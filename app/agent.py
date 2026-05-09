"""
LangGraph agent for query reformulation and re-retrieval
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time

from langgraph.graph import StateGraph, END
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.retrieval import RetrievalQAChain, AdvancedRetriever, RetrievalResult, QAResult
from app.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentState:
    """State threaded through the LangGraph workflow nodes."""
    query: str
    original_query: str
    org_id: str
    top_k: int
    confidence_threshold: float
    max_attempts: int
    current_attempt: int = 1

    retrieval_results: Optional[List[RetrievalResult]] = None
    qa_result: Optional[QAResult] = None

    reformulation_reason: Optional[str] = None
    keywords_extracted: Optional[List[str]] = None
    metadata_filters: Optional[Dict[str, Any]] = None

    attempts: List[Dict[str, Any]] = field(default_factory=list)


class QueryReformulator:
    """Handles query reformulation strategies."""

    def __init__(self, openai_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        # Groq exposes an OpenAI-compatible endpoint — no other change needed
        self._client = OpenAI(
            api_key=openai_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def extract_keywords_from_chunks(
        self, chunks: List[RetrievalResult], max_keywords: int = 5
    ) -> List[str]:
        """Extract important keywords from the top retrieved chunks."""
        try:
            combined_text = " ".join([c.text for c in chunks[:3]])
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the most important keywords from the given text. "
                            "Return only the keywords separated by commas, no explanations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Extract up to {max_keywords} important keywords:\n\n{combined_text}"
                        ),
                    },
                ],
                max_tokens=100,
                temperature=0.1,
            )
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
            logger.info("Keywords extracted", keywords=keywords, chunk_count=len(chunks))
            return keywords[:max_keywords]
        except Exception as e:
            logger.error("Failed to extract keywords", error=str(e))
            return []

    def extract_metadata_filters(self, chunks: List[RetrievalResult]) -> Dict[str, Any]:
        """Extract common metadata filters from top retrieved chunks."""
        try:
            from collections import Counter

            all_metadata: Dict[str, list] = {}
            for chunk in chunks[:3]:
                if chunk.metadata:
                    for key, value in chunk.metadata.items():
                        all_metadata.setdefault(key, []).append(value)

            filters = {}
            for key, values in all_metadata.items():
                if len(values) >= 2:
                    filters[key] = Counter(values).most_common(1)[0][0]

            logger.info("Metadata filters extracted", filters=filters)
            return filters
        except Exception as e:
            logger.error("Failed to extract metadata filters", error=str(e))
            return {}

    def reformulate_query(
        self,
        original_query: str,
        keywords: List[str],
        metadata_filters: Dict[str, Any],
        confidence: float,
    ) -> str:
        """Reformulate a low-confidence query using extracted context."""
        try:
            context_parts = []
            if keywords:
                context_parts.append(f"Important keywords: {', '.join(keywords)}")
            if metadata_filters:
                filter_str = ", ".join(f"{k}: {v}" for k, v in metadata_filters.items())
                context_parts.append(f"Relevant metadata: {filter_str}")
            context = "\n".join(context_parts) or "No additional context available."

            prompt = (
                f"The original query had low confidence ({confidence:.2f}). "
                f"Reformulate it to be more specific and include relevant keywords.\n\n"
                f"Original query: {original_query}\n"
                f"Additional context: {context}\n\n"
                f"Reformulated query:"
            )
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a query reformulation expert. Create a more specific "
                            "version of the original query that incorporates relevant "
                            "keywords and context."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            reformulated = response.choices[0].message.content.strip()
            logger.info(
                "Query reformulated",
                original_query=original_query,
                reformulated_query=reformulated,
                confidence=confidence,
            )
            return reformulated
        except Exception as e:
            logger.error("Failed to reformulate query", error=str(e))
            return f"{original_query} {' '.join(keywords[:2])}" if keywords else original_query


class IntelligentQAAgent:
    """LangGraph agent for intelligent question answering with query reformulation."""

    def __init__(
        self,
        session,
        org_id: str,
        openai_api_key: str,
        confidence_threshold: float = 0.7,
        max_attempts: int = 2,
        model_name: str = "llama-3.3-70b-versatile",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self.session = session
        self.org_id = org_id
        self.openai_api_key = openai_api_key
        self.confidence_threshold = confidence_threshold
        self.max_attempts = max_attempts
        self.model_name = model_name
        self._embedding_model = embedding_model

        self.qa_chain = RetrievalQAChain(
            session=session,
            org_id=org_id,
            openai_api_key=openai_api_key,
            model_name=model_name,
        )
        self.reformulator = QueryReformulator(openai_api_key, model_name)
        self.graph = self._build_graph()

    # ── Graph construction ─────────────────────────────────────────────────

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("initial_retrieval", self._initial_retrieval)
        workflow.add_node("check_confidence", self._check_confidence)
        workflow.add_node("extract_context", self._extract_context)
        workflow.add_node("reformulate_query", self._reformulate_query)
        workflow.add_node("re_retrieval", self._re_retrieval)
        workflow.add_node("compare_results", self._compare_results)

        workflow.set_entry_point("initial_retrieval")
        workflow.add_edge("initial_retrieval", "check_confidence")
        workflow.add_conditional_edges(
            "check_confidence",
            self._should_reformulate,
            {"reformulate": "extract_context", "accept": END},
        )
        workflow.add_edge("extract_context", "reformulate_query")
        workflow.add_edge("reformulate_query", "re_retrieval")
        workflow.add_edge("re_retrieval", "compare_results")
        workflow.add_edge("compare_results", END)

        return workflow.compile()

    # ── Node implementations ───────────────────────────────────────────────

    async def _initial_retrieval(self, state: AgentState) -> AgentState:
        logger.info("Starting initial retrieval", query=state.query, attempt=state.current_attempt)
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self._embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            query_embedding = embeddings.embed_query(state.query)
            qa_result = await self.qa_chain.answer_question(
                query=state.query, query_embedding=query_embedding, top_k=state.top_k
            )
            state.qa_result = qa_result
            state.attempts.append(
                {
                    "attempt": state.current_attempt,
                    "query": state.query,
                    "answer": qa_result.answer,
                    "confidence": qa_result.confidence,
                    "sources_count": len(qa_result.sources),
                    "processing_time": qa_result.processing_time,
                    "total_tokens": qa_result.total_tokens,
                }
            )
            logger.info(
                "Initial retrieval completed",
                confidence=qa_result.confidence,
                sources_count=len(qa_result.sources),
            )
        except Exception as e:
            logger.error("Initial retrieval failed", error=str(e))
            state.qa_result = QAResult(
                answer="Failed to process query",
                confidence=0.0,
                sources=[],
                total_tokens=0,
                processing_time=0.0,
            )
        return state

    def _should_reformulate(self, state: AgentState) -> str:
        if not state.qa_result:
            return "accept"
        should = (
            state.qa_result.confidence < state.confidence_threshold
            and state.current_attempt < state.max_attempts
        )
        logger.info(
            "Confidence check",
            confidence=state.qa_result.confidence,
            threshold=state.confidence_threshold,
            should_reformulate=should,
        )
        return "reformulate" if should else "accept"

    # _check_confidence is a passthrough node (routing is done by conditional edges above)
    async def _check_confidence(self, state: AgentState) -> AgentState:
        return state

    async def _extract_context(self, state: AgentState) -> AgentState:
        logger.info("Extracting context for reformulation")
        try:
            if state.qa_result and state.qa_result.sources:
                proxy_chunks = [
                    RetrievalResult(
                        chunk_id=s["chunk_id"],
                        document_id=s["document_id"],
                        text=s["text_preview"],
                        score=s["score"],
                        metadata={},
                        source_title=s["title"],
                        source_url=s["url"],
                    )
                    for s in state.qa_result.sources
                ]
                state.keywords_extracted = self.reformulator.extract_keywords_from_chunks(
                    proxy_chunks
                )
                state.metadata_filters = self.reformulator.extract_metadata_filters(proxy_chunks)
                state.reformulation_reason = f"Low confidence: {state.qa_result.confidence:.2f}"
        except Exception as e:
            logger.error("Failed to extract context", error=str(e))
            state.keywords_extracted = []
            state.metadata_filters = {}
        return state

    async def _reformulate_query(self, state: AgentState) -> AgentState:
        logger.info("Reformulating query")
        try:
            state.query = self.reformulator.reformulate_query(
                original_query=state.original_query,
                keywords=state.keywords_extracted or [],
                metadata_filters=state.metadata_filters or {},
                confidence=state.qa_result.confidence if state.qa_result else 0.0,
            )
            state.current_attempt += 1
        except Exception as e:
            logger.error("Failed to reformulate query", error=str(e))
        return state

    async def _re_retrieval(self, state: AgentState) -> AgentState:
        logger.info("Performing re-retrieval", query=state.query, attempt=state.current_attempt)
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self._embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            query_embedding = embeddings.embed_query(state.query)
            qa_result = await self.qa_chain.answer_question(
                query=state.query, query_embedding=query_embedding, top_k=state.top_k
            )
            state.qa_result = qa_result
            state.attempts.append(
                {
                    "attempt": state.current_attempt,
                    "query": state.query,
                    "answer": qa_result.answer,
                    "confidence": qa_result.confidence,
                    "sources_count": len(qa_result.sources),
                    "processing_time": qa_result.processing_time,
                    "total_tokens": qa_result.total_tokens,
                    "reformulation_reason": state.reformulation_reason,
                    "keywords_used": state.keywords_extracted,
                    "metadata_filters": state.metadata_filters,
                }
            )
        except Exception as e:
            logger.error("Re-retrieval failed", error=str(e))
        return state

    async def _compare_results(self, state: AgentState) -> AgentState:
        if len(state.attempts) <= 1:
            return state
        best = max(state.attempts, key=lambda x: x["confidence"])
        if state.qa_result and best["confidence"] > state.qa_result.confidence:
            state.qa_result = QAResult(
                answer=best["answer"],
                confidence=best["confidence"],
                sources=[],
                total_tokens=best["total_tokens"],
                processing_time=best["processing_time"],
            )
        logger.info(
            "Results compared",
            attempts_count=len(state.attempts),
            best_confidence=best["confidence"],
        )
        return state

    # ── Public entry point ─────────────────────────────────────────────────

    async def answer_question(
        self,
        query: str,
        top_k: int = 5,
        confidence_threshold: float = None,
        max_attempts: int = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        confidence_threshold = confidence_threshold or self.confidence_threshold
        max_attempts = max_attempts or self.max_attempts

        logger.info(
            "Starting intelligent QA agent",
            query=query[:100],
            confidence_threshold=confidence_threshold,
            max_attempts=max_attempts,
        )
        try:
            state = AgentState(
                query=query,
                original_query=query,
                org_id=self.org_id,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
                max_attempts=max_attempts,
            )
            final_state = await self.graph.ainvoke(state)

            response = {
                "query": query,
                "final_answer": (
                    final_state.qa_result.answer if final_state.qa_result else "No answer generated"
                ),
                "final_confidence": (
                    final_state.qa_result.confidence if final_state.qa_result else 0.0
                ),
                "total_attempts": len(final_state.attempts),
                "attempts": final_state.attempts,
                "reformulation_used": len(final_state.attempts) > 1,
                "processing_time": time.time() - start_time,
            }
            if final_state.reformulation_reason:
                response["reformulation_reason"] = final_state.reformulation_reason
            if final_state.keywords_extracted:
                response["keywords_extracted"] = final_state.keywords_extracted
            if final_state.metadata_filters:
                response["metadata_filters"] = final_state.metadata_filters

            logger.info(
                "Intelligent QA agent completed",
                final_confidence=response["final_confidence"],
                total_attempts=response["total_attempts"],
            )
            return response

        except Exception as e:
            logger.error("Intelligent QA agent failed", error=str(e), exc_info=True)
            return {
                "query": query,
                "final_answer": "Failed to process query",
                "final_confidence": 0.0,
                "total_attempts": 0,
                "attempts": [],
                "reformulation_used": False,
                "processing_time": time.time() - start_time,
                "error": str(e),
            }
