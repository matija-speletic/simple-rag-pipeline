from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import BaseNode
from itertools import pairwise
import multiprocessing as mp
from document_chunk import Chunk


# reader = SimpleDirectoryReader(input_dir="data")
# documentss = reader.load_data()


def overlap_documents(documents: list[Document],
                      overlap_ratio=0.15,
                      delimiters=(".", "!", "?", "\n")):
    for (prev_doc, next_doc) in pairwise(documents):
        text = next_doc.text
        text_length = len(text)
        overlap_length = int(text_length * overlap_ratio)
        overlap_text = text[:overlap_length]
        split_index = max([overlap_text.rfind(delimiter) for delimiter in delimiters])
        prev_doc.text += text[:split_index]


# pipeline = IngestionPipeline(
#     transformations=[
#         SentenceSplitter(chunk_size=200, chunk_overlap=40),
#     ],
# )

# nodes = pipeline.run(documents=[Document.example()])
#
# print(nodes[0])

# class Chunk:
#     def __init__(self, node: BaseNode):
#         self.text: str = node.text
#         self.document: str = node.metadata["file_name"]
#         self.date_created: str = node.metadata["creation_date"]
#         self.date_modified: str = node.metadata["last_modified_date"]


class DocumentSplitter:
    def __init__(self, chunk_size=200, chunk_overlap=40):
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            ],
        )
        self.documents: list[Document] = []

    def load_directory(self, directory):
        reader = SimpleDirectoryReader(input_dir=directory)
        self.documents += reader.load_data()

    def load_file(self, file):
        reader = SimpleDirectoryReader(input_files=[file])
        self.documents += reader.load_data()

    def split(self) -> list[Chunk]:
        nodes = self.pipeline.run(
            documents=self.documents)
        return [Chunk.from_llama_index_node(node) for node in nodes]


if __name__ == "__main__":
    ds = DocumentSplitter()
    ds.load_directory("data")
    print(len(ds.documents))
    print(ds.documents[0].metadata)
    print(ds.documents[0])
    print(ds.split())
