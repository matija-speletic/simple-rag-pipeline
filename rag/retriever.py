from pathlib import Path

from rag.document_loader import DocumentLoader
from llm.base import EmbeddingModel
from models import DocumentChunk
from vector_store.base import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore,
                 embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str,
                 num_chunks: int = 5) -> list[DocumentChunk]:
        query_embedding = self.embedding_model.get_query_embedding(query)
        ranked_chunks = self.vector_store.retrieve(query_embedding, num_chunks)
        return [chunk for chunk, _ in ranked_chunks]

    def list_data_sources(self):
        return self.vector_store.list_documents()

    def load_data_sources(self, paths: Path | list[Path],
                          overlap_pages: bool = False,
                          overlap_ratio: float = 0.15,
                          reset_data_sources: bool = False):
        loader = DocumentLoader()
        if isinstance(paths, Path):
            paths = [paths]
        for path in paths:
            if path.is_dir():
                loader.load_directory(path)
            else:
                loader.load_file(path)
        if overlap_pages:
            loader.overlap_pages(overlap_ratio)

        chunks = loader.split()
        chunk_embeddings = self.embedding_model.get_text_embedding_batch(
            [chunk.text for chunk in chunks])
        for chunk, embedding in zip(chunks, chunk_embeddings):
            chunk.embedding = embedding

        if reset_data_sources:
            self.vector_store.clear_data()
        self.vector_store.add_batch_chunks(chunks)
