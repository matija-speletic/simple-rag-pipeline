from abc import ABC, abstractmethod

from document_chunk import Chunk


class VectorStore(ABC):
    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def create_index(self):
        pass

    @abstractmethod
    def add_batch_chunks(self, chunks: list[Chunk]):
        pass

    @abstractmethod
    def retrieve(self,
                 embedding: list[float],
                 nearest_neighbors: int = 5) -> list[tuple[Chunk, float]]:
        pass

    def clear_data(self):
        pass

