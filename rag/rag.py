from llama_index.core.llms import MessageRole

from llm.base import LLM
from models import DocumentChunk
from rag.retriever import Retriever


class RAG(Retriever):
    def __init__(self, vector_store, embedding_model,
                 llm: LLM):
        super().__init__(vector_store, embedding_model)
        self.llm = llm

    def generate(
            self, prompt: str,
            history: list[tuple[MessageRole, str]],
            stream: bool = False,
            use_rag: bool = True
    ) -> tuple[str, list[DocumentChunk]]:
        context = None
        chunks = []
        if use_rag:
            chunks = self.retrieve(prompt, num_chunks=3)
            context = ""
            for chunk in chunks:
                context += "Piece of context from document: " + chunk.document_name + "\n"
                context += chunk.text + "\n\n"
        response = self.llm.generate(prompt, history, context, stream)
        return response, chunks
