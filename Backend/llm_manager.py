import logging
from typing import List, Optional
from langchain_core.documents import Document
from groq import Groq

logger = logging.getLogger("enhanced_rag")

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

    def generate_response(self, query: str, context_docs: List[Document], model_name: Optional[str] = None) -> str:
        try:
            if not self.client:
                self.initialize()

            context_sections = []
            for i, doc in enumerate(context_docs):
                metadata = doc.metadata
                source = metadata.get("filename", "Unknown")
                file_type = metadata.get("file_type", "document")
                chunk = metadata.get("chunk", "")
                total_chunks = metadata.get("total_chunks", "")
                chunk_info = f" (Chunk {chunk}/{total_chunks})" if chunk and total_chunks else ""

                section = f"[DOCUMENT {i+1}]\nSource: {source}{chunk_info}\nType: {file_type}\nContent:\n{doc.page_content}\n"
                context_sections.append(section)

            context = "\n".join(context_sections)

            system_prompt = (
                "You are a helpful and precise assistant that provides accurate information based on the context provided. "
                "Follow these rules when generating answers:\n"
                "1. Only use information from the provided context documents.\n"
                "2. If the context doesn't contain information needed to fully answer the question, say so clearly.\n"
                "3. Do not invent or assume information that's not in the context.\n"
                "4. Cite the specific document sources when providing information.\n"
                "5. Format your answer for clarity and readability.\n"
                "6. If different documents contain conflicting information, acknowledge this and explain the differences."
            )

            user_prompt = (
                f"Question: {query}\n\n"
                f"Context Documents:\n{context}\n\n"
                f"Please provide a comprehensive answer based solely on the provided context documents. "
                f"Cite specific documents when providing information."
            )

            chosen_model = model_name or self.config.groq_model

            response = self.client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=8192,
            )

            answer = response.choices[0].message.content
            logger.info(f"Generated response for query: {query[:50]}...")
            return answer
        except Exception as e:
            logger.error(f"Response error: {str(e)}")
            return "An error occurred while generating the answer."

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
                    user_prompt = (
                        f"Query: {query}\n\n"
                        f"Document content:\n{doc.page_content}\n\n"
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
                        relevance_score = float(response.choices[0].message.content.strip())
                    except ValueError:
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
