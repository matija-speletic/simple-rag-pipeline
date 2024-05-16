from llama_index.core.llms import MessageRole

from llm.base import LLM
from models import DocumentChunk
from rag.retriever import Retriever


class RAG(Retriever):
    def __init__(self, vector_store, embedding_model,
                 llm: LLM | None = None,
                 num_chunks: int = 5):
        super().__init__(vector_store, embedding_model)
        self.llm = llm

        self.num_chunks = num_chunks

    def _prepare_inputs(self, prompt: str,
                        use_rag: bool) -> tuple[str, list[DocumentChunk]]:
        context = None
        chunks = []
        if use_rag:
            chunks = self.retrieve(prompt, num_chunks=self.num_chunks)
            context = ""
            for chunk in chunks:
                context += "Piece of context from document: " + chunk.document_name + "\n"
                context += chunk.text + "\n\n"
        return context, chunks

    def generate(
            self, prompt: str,
            history: list[tuple[MessageRole, str]],
            stream: bool = False,
            use_rag: bool = True
    ) -> tuple[str, list[DocumentChunk]]:
        context, chunks = self._prepare_inputs(prompt, use_rag)
        if self.llm is None:
            response = ""
        else:
            response = self.llm.generate(prompt, history, context)
        return response, chunks

    def generate_stream(
            self, prompt: str,
            history: list[tuple[MessageRole, str]],
            use_rag: bool = True
    ):
        context, chunks = self._prepare_inputs(prompt, use_rag)

        stream = self.llm.generate_stream(prompt, history, context)
        return stream, chunks
