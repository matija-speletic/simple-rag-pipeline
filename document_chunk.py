from llama_index.core.schema import BaseNode


class Chunk:
    def __init__(self, text: str, document_name: str,
                 document_path: str,
                 page: str, embedding: list[float]):
        self.text: str = text
        self.document_name: str = document_name
        self.document_path: str = document_path
        self.page: str = page
        self.embedding: list[float] = embedding

    @classmethod
    def from_llama_index_node(cls, node: BaseNode):
        return cls(
            node.text,
            node.metadata["file_name"],
            node.metadata["file_path"],
            node.metadata["page_label"],
            [])
