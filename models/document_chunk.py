from llama_index.core.schema import BaseNode
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    text: str
    document_name: str
    document_path: str
    page: str
    embedding: list[float]

    @classmethod
    def from_llama_index_node(cls, node: BaseNode):
        return cls(
            node.text,
            node.metadata["file_name"],
            node.metadata["file_path"],
            node.metadata["page_label"],
            [])
