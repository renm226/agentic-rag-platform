"""
CrewAI multi-agent pipeline for complex RAG queries.

Four sequential agents:
  1. Query Planner       — decomposes the question, identifies key concepts
  2. Retrieval Specialist — curates the most relevant chunks, searches for more if needed
  3. Answer Synthesizer   — builds a cited, structured answer from the evidence
  4. Fact Checker         — verifies claims against sources, scores confidence

The crew runs synchronously (crew.kickoff) and is called from async FastAPI
endpoints via asyncio.to_thread so it never blocks the event loop.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import httpx
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

from app.logging import get_logger

logger = get_logger(__name__)


# ── Retrieval search tool (optional) ──────────────────────────────────────────

def _make_search_tool(api_base_url: str, org_id: str):
    """Return a crewai Tool that hits /retrieve for targeted re-retrieval."""

    @tool("Search Documents")
    def search_documents(query: str) -> str:
        """Search for additional relevant document chunks when initial context is insufficient.

        Args:
            query: A specific search query string targeting the missing information.

        Returns:
            Ranked document passages with source titles and relevance scores.
        """
        try:
            resp = httpx.post(
                f"{api_base_url}/retrieve",
                params={"query": query, "top_k": 5, "org_id": org_id},
                timeout=15.0,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return "No additional documents found for this query."

            parts = []
            for i, r in enumerate(results, 1):
                parts.append(
                    f"[Result {i} | Score: {r.get('score', 0):.3f}]\n"
                    f"Source: {r.get('source_title', 'Unknown')}\n"
                    f"Text: {r.get('text', '')}"
                )
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.error("Search tool request failed", error=str(e))
            return f"Search unavailable: {e}"

    return search_documents


# ── RAGCrew ───────────────────────────────────────────────────────────────────

class RAGCrew:
    """
    CrewAI multi-agent pipeline that answers complex queries over pre-retrieved
    document chunks.

    Usage (from async code):
        crew = RAGCrew(openai_api_key=..., model_name=..., api_base_url=...)
        result = await asyncio.to_thread(crew.run, query=q, retrieved_chunks=chunks)
    """

    def __init__(
        self,
        openai_api_key: str,           # holds the xAI API key
        model_name: str = "grok-beta",
        api_base_url: Optional[str] = None,
        org_id: str = "default",
        verbose: bool = False,
    ):
        # xAI is OpenAI-compatible; LiteLLM routes via base_url
        self._llm = LLM(
            model=f"openai/{model_name}",
            api_key=openai_api_key,
            base_url="https://api.x.ai/v1",
        )
        self._api_base_url = api_base_url
        self._org_id = org_id
        self._verbose = verbose

    # ── Public entry point ────────────────────────────────────────────────

    def run(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute the four-agent pipeline and return a structured result dict."""
        start_time = time.time()
        context = self._format_context(retrieved_chunks)

        query_planner, retrieval_specialist, answer_synthesizer, fact_checker = (
            self._build_agents()
        )

        analyze_task = Task(
            description=(
                f"Analyze this user query thoroughly:\n\n"
                f"QUERY: {query}\n\n"
                f"Produce a structured analysis containing:\n"
                f"1. Core information need — what the user truly wants to know\n"
                f"2. Sub-questions — list every distinct question that must be answered\n"
                f"3. Key concepts and entities to look for in documents\n"
                f"4. Expected answer format (factual, explanatory, comparative, procedural, etc.)"
            ),
            expected_output=(
                "Structured analysis: core intent statement, numbered sub-questions, "
                "key concepts list, and expected format"
            ),
            agent=query_planner,
        )

        retrieve_task = Task(
            description=(
                f"Select the most relevant evidence from the pre-retrieved documents "
                f"to answer the query.\n\n"
                f"RETRIEVED DOCUMENTS:\n{context}\n\n"
                f"Steps:\n"
                f"1. Rate each document High / Medium / Low relevance\n"
                f"2. Keep the 5–7 highest-relevance passages\n"
                f"3. If critical information is missing, use the Search Documents tool "
                f"to retrieve it\n"
                f"4. Group selected passages by sub-question or topic\n\n"
                f"Output the curated evidence with [Source: title] attribution for each passage."
            ),
            expected_output=(
                "Curated evidence list: each entry includes source title, relevance rating, "
                "and the key passage text"
            ),
            agent=retrieval_specialist,
        )

        synthesize_task = Task(
            description=(
                f"Write a comprehensive answer using the curated evidence.\n\n"
                f"ORIGINAL QUERY: {query}\n\n"
                f"Requirements:\n"
                f"1. Answer the query directly and completely\n"
                f"2. Address every sub-question from the analysis\n"
                f"3. Cite sources as [Source: title] after each claim\n"
                f"4. Use clear prose structure; use bullets only when listing items\n"
                f"5. Explicitly state when information is incomplete or uncertain"
            ),
            expected_output=(
                "Comprehensive, well-cited answer that addresses all sub-questions "
                "with clear source attribution"
            ),
            agent=answer_synthesizer,
        )

        verify_task = Task(
            description=(
                f"Verify the synthesized answer against the source documents.\n\n"
                f"ORIGINAL QUERY: {query}\n\n"
                f"Checks:\n"
                f"1. Every claim is supported by at least one retrieved document\n"
                f"2. No contradictions between sources\n"
                f"3. No important information is missing\n\n"
                f"You MUST respond in EXACTLY this format (preserve the labels):\n"
                f"VERIFIED ANSWER: <final answer, corrected if needed>\n"
                f"CONFIDENCE: <decimal 0.0–1.0>\n"
                f"ISSUES: <bulleted list of unsupported claims, or 'None'>\n"
                f"SOURCES_USED: <bulleted list of source titles that back the answer>"
            ),
            expected_output=(
                "Output with four labeled sections: VERIFIED ANSWER, CONFIDENCE, "
                "ISSUES, SOURCES_USED"
            ),
            agent=fact_checker,
        )

        crew = Crew(
            agents=[query_planner, retrieval_specialist, answer_synthesizer, fact_checker],
            tasks=[analyze_task, retrieve_task, synthesize_task, verify_task],
            process=Process.sequential,
            verbose=self._verbose,
        )

        crew_output = crew.kickoff()
        processing_time = time.time() - start_time
        parsed = self._parse_output(str(crew_output))

        logger.info(
            "CrewAI pipeline completed",
            query=query[:80],
            confidence=parsed["confidence"],
            issues_count=len(parsed["issues"]),
            processing_time=round(processing_time, 2),
        )

        return {
            "query": query,
            "answer": parsed["answer"],
            "confidence": parsed["confidence"],
            "issues": parsed["issues"],
            "sources_used": parsed["sources_used"],
            "agents_used": [
                "query_planner",
                "retrieval_specialist",
                "answer_synthesizer",
                "fact_checker",
            ],
            "processing_time": processing_time,
        }

    # ── Agent construction ────────────────────────────────────────────────

    def _build_agents(self) -> tuple:
        tools = (
            [_make_search_tool(self._api_base_url, self._org_id)]
            if self._api_base_url
            else []
        )

        query_planner = Agent(
            role="Query Analysis Specialist",
            goal=(
                "Analyze complex user queries, identify the core information need, "
                "and decompose multi-part questions into clear, answerable sub-questions."
            ),
            backstory=(
                "You are an expert at understanding what users truly need from a document "
                "collection. You excel at recognising implicit requirements, breaking down "
                "compound questions, and identifying the key concepts that retrieval must cover."
            ),
            llm=self._llm,
            allow_delegation=False,
            verbose=self._verbose,
        )

        retrieval_specialist = Agent(
            role="Document Retrieval Expert",
            goal=(
                "Select the most relevant evidence from retrieved documents and, "
                "when necessary, search for additional targeted information."
            ),
            backstory=(
                "You are an expert at evaluating source relevance and identifying which "
                "passages genuinely support an answer versus which are tangential. "
                "You know when to search for more specific information and how to formulate "
                "precise retrieval queries."
            ),
            tools=tools,
            llm=self._llm,
            allow_delegation=False,
            verbose=self._verbose,
        )

        answer_synthesizer = Agent(
            role="Knowledge Synthesis Expert",
            goal=(
                "Produce comprehensive, well-structured answers by combining evidence "
                "from multiple document sources with clear citation."
            ),
            backstory=(
                "You are an expert at synthesising information from disparate sources into "
                "clear, accurate prose. You cite every factual claim, present information "
                "in logical order, and candidly acknowledge gaps or uncertainty."
            ),
            llm=self._llm,
            allow_delegation=False,
            verbose=self._verbose,
        )

        fact_checker = Agent(
            role="Accuracy Verification Specialist",
            goal=(
                "Verify every claim in the synthesised answer against the source documents, "
                "flag unsupported assertions, and assign a calibrated confidence score."
            ),
            backstory=(
                "You are an expert at cross-referencing claims against primary sources. "
                "You catch statements that go beyond what the evidence supports, surface "
                "contradictions between sources, and produce honest confidence estimates "
                "based on evidence quality and completeness."
            ),
            llm=self._llm,
            allow_delegation=False,
            verbose=self._verbose,
        )

        return query_planner, retrieval_specialist, answer_synthesizer, fact_checker

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _format_context(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No documents retrieved."
        parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("source_title") or "Unknown"
            url = chunk.get("source_url") or ""
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")
            source_line = f"Source: {title}" + (f" | {url}" if url else "")
            parts.append(f"[Doc {i} | Relevance: {score:.3f}]\n{source_line}\n{text}")
        sep = "\n\n" + "─" * 60 + "\n\n"
        return sep.join(parts)

    @staticmethod
    def _parse_output(output: str) -> Dict[str, Any]:
        """Extract structured fields from the fact-checker's output."""
        result: Dict[str, Any] = {
            "answer": output.strip(),
            "confidence": 0.7,
            "issues": [],
            "sources_used": [],
        }

        answer_match = re.search(
            r"VERIFIED ANSWER:\s*(.*?)(?=\nCONFIDENCE:|\Z)", output, re.DOTALL | re.IGNORECASE
        )
        if answer_match:
            result["answer"] = answer_match.group(1).strip()

        confidence_match = re.search(
            r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", output, re.IGNORECASE
        )
        if confidence_match:
            try:
                result["confidence"] = min(1.0, max(0.0, float(confidence_match.group(1))))
            except ValueError:
                pass

        issues_match = re.search(
            r"ISSUES:\s*(.*?)(?=\nSOURCES_USED:|\Z)", output, re.DOTALL | re.IGNORECASE
        )
        if issues_match:
            text = issues_match.group(1).strip()
            if text.lower() not in ("none", "no issues", "none found", ""):
                result["issues"] = [
                    ln.lstrip("•-* ").strip() for ln in text.splitlines() if ln.strip()
                ]

        sources_match = re.search(
            r"SOURCES_USED:\s*(.*?)$", output, re.DOTALL | re.IGNORECASE
        )
        if sources_match:
            text = sources_match.group(1).strip()
            result["sources_used"] = [
                ln.lstrip("•-* ").strip() for ln in text.splitlines() if ln.strip()
            ]

        return result
