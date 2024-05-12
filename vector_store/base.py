from abc import ABC, abstractmethod

from models import DocumentChunk


class VectorStore(ABC):
    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def create_index(self):
        pass

    @abstractmethod
    def add_batch_chunks(self, chunks: list[DocumentChunk]):
        pass

    @abstractmethod
    def retrieve(self,
                 embedding: list[float],
                 nearest_neighbors: int = 5) -> list[tuple[DocumentChunk, float]]:
        pass

    @abstractmethod
    def clear_data(self):
        pass

    @abstractmethod
    def list_documents(self) -> list[str]:
        pass

