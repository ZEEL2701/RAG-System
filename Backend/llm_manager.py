import logging
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from groq import Groq

logger = logging.getLogger("enhanced_rag")


def _truncate_context_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _is_token_or_rate_error(err: Exception) -> bool:
    t = str(err).lower()
    return any(
        x in t
        for x in ("413", "429", "rate_limit", "tokens per minute", "too many tokens", " tpm", "token")
    )


class LLMManager:
    def __init__(self, config):
        self.config = config
        self.client = None

    def initialize(self):
        try:
            if not self.config.groq_api_key:
                logger.error("Groq API key missing")
                raise ValueError("Groq API key required")
            self.client = Groq(
                api_key=self.config.groq_api_key,
                base_url=self.config.groq_base_url
            )
            logger.info(f"Groq client initialized with model: {self.config.groq_model}")
            try:
                models = self.client.models.list()
                logger.info(f"Successfully connected to Groq API. Models available.")
            except Exception as e:
                logger.error(f"Failed to list Groq models, continuing: {str(e)}")
        except Exception as e:
            logger.error(f"LLM initialization error: {str(e)}")
            raise

    def _extractive_fallback(self, query: str, context_docs: List[Document], error_note: Optional[str] = None) -> str:
        if not context_docs:
            return (
                "No model answer and no retrieved context."
                + (f" ({error_note})" if error_note else "")
            )
        parts = [
            "The model did not return a summary (often a Groq rate/token limit on the free tier, or try again in a minute). "
            "Below is the text the retriever found for your question — this is the same content as in Sources.",
        ]
        if error_note:
            parts.append(f"Detail: {error_note}")
        parts.append("")
        for i, doc in enumerate(context_docs[:4], 1):
            name = doc.metadata.get("filename", "Unknown")
            excerpt = _truncate_context_text(doc.page_content, 900)
            parts.append(f"--- {i}. {name} ---\n{excerpt}\n")
        return "\n".join(parts).strip()

    def _build_messages(
        self, query: str, docs: List[Document], char_limit: int
    ) -> Tuple[str, str]:
        context_sections = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            source = metadata.get("filename", "Unknown")
            file_type = metadata.get("file_type", "document")
            chunk = metadata.get("chunk", "")
            total_chunks = metadata.get("total_chunks", "")
            chunk_info = f" (Chunk {chunk}/{total_chunks})" if chunk and total_chunks else ""
            body = _truncate_context_text(doc.page_content, char_limit)
            section = f"[{i+1}] {source}{chunk_info} ({file_type})\n{body}"
            context_sections.append(section)
        context = "\n\n".join(context_sections)
        system_prompt = (
            "Answer using only the context below. If it is not enough, say what is missing. "
            "Be concise; cite the source filename."
        )
        user_prompt = f"Question: {query}\n\nContext:\n{context}"
        return system_prompt, user_prompt

    def generate_response(self, query: str, context_docs: List[Document], model_name: Optional[str] = None) -> str:
        chosen_model = model_name or self.config.groq_model

        if not context_docs:
            try:
                if not self.client:
                    self.initialize()
                response = self.client.chat.completions.create(
                    model=chosen_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful, concise assistant."},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.3,
                    max_tokens=self.config.groq_completion_max_tokens,
                )
                raw = getattr(response.choices[0].message, "content", None)
                out = (raw or "").strip()
                return out or "The model returned an empty reply."
            except Exception as e:
                logger.error(f"LLM-only response error: {e}")
                return f"Could not get a model reply: {e}"
        attempts: List[Tuple[List[Document], int, int]] = [
            (context_docs, self.config.max_chars_per_context_doc, self.config.groq_completion_max_tokens),
        ]
        if len(context_docs) > 1:
            attempts.append(
                (context_docs[:2], min(500, self.config.max_chars_per_context_doc), min(384, self.config.groq_completion_max_tokens))
            )

        try:
            if not self.client:
                self.initialize()

            for attempt_idx, (doc_slice, char_limit, max_out) in enumerate(attempts):
                system_prompt, user_prompt = self._build_messages(query, doc_slice, char_limit)
                try:
                    response = self.client.chat.completions.create(
                        model=chosen_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.3,
                        max_tokens=max_out,
                    )
                    msg = response.choices[0].message
                    raw = getattr(msg, "content", None)
                    answer = (raw or "").strip()
                    if answer:
                        logger.info(f"Generated response for query: {query[:50]}...")
                        return answer
                    logger.warning(f"Empty content from model {chosen_model}")
                    return self._extractive_fallback(query, context_docs, "Model returned empty content.")
                except Exception as e:
                    logger.error(f"Response error (docs={len(doc_slice)}, char_limit={char_limit}): {e}")
                    if _is_token_or_rate_error(e) and attempt_idx + 1 < len(attempts):
                        continue
                    return self._extractive_fallback(query, context_docs, str(e))
        except Exception as e:
            logger.error(f"LLM setup or fatal error: {e}")
            return self._extractive_fallback(query, context_docs, str(e))

    def rank_relevance(self, query: str, docs_with_scores: List[tuple]) -> List[Document]:
        try:
            if not self.client:
                self.initialize()
            if not docs_with_scores:
                return []

            threshold_score = 0.3  # Adjust as needed
            filtered_docs = [(doc, score) for doc, score in docs_with_scores if score < threshold_score]

            if not filtered_docs:
                filtered_docs = sorted(docs_with_scores, key=lambda x: x[1])[:3]

            if len(filtered_docs) > self.config.max_context_documents:
                ranked_docs = []
                for doc, score in filtered_docs:
                    system_prompt = (
                        "You are a document relevance ranker. Your task is to score how relevant a document is to a query.\n"
                        "Rate the relevance on a scale of 0-10, where 10 is perfectly relevant and 0 is completely irrelevant.\n"
                        "Your response must be exactly one number between 0 and 10."
                    )
                    doc_excerpt = _truncate_context_text(doc.page_content, self.config.max_chars_per_context_doc)
                    user_prompt = (
                        f"Query: {query}\n\n"
                        f"Document content:\n{doc_excerpt}\n\n"
                        f"Rate the relevance of this document to the query on a scale of 0-10:"
                    )

                    response = self.client.chat.completions.create(
                        model=self.config.groq_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=self.config.max_tokens,
                    )

                    try:
                        raw_rank = response.choices[0].message.content
                        relevance_score = float((raw_rank or "").strip())
                    except (ValueError, TypeError, AttributeError):
                        relevance_score = 5.0

                    ranked_docs.append((doc, relevance_score))

                sorted_docs = sorted(ranked_docs, key=lambda x: x[1], reverse=True)
                result = [doc for doc, _ in sorted_docs[:self.config.max_context_documents]]
                logger.info(f"LLM ranked {len(filtered_docs)} docs; using top {len(result)} relevant docs.")
                return result
            else:
                return [doc for doc, _ in filtered_docs]
        except Exception as e:
            logger.error(f"Ranking error: {str(e)}")
            return [doc for doc, _ in docs_with_scores[:self.config.max_context_documents]]
