from itertools import pairwise

from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from models.document_chunk import DocumentChunk


class DocumentLoader:
    def __init__(self, chunk_size=300, chunk_overlap=60):
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

    def overlap_pages(self, overlap_ratio=0.15, delimiters=(".", "!", "?", "\n")):
        page_pairs = list(filter(
            lambda p: p[0].metadata["file_name"] == p[1].metadata["file_name"]
            and p[0].metadata["page_label"] != p[1].metadata["page_label"],
            pairwise(self.documents)))

        for (page1, page2) in page_pairs:
            text1 = page1.text
            text2 = page2.text
            text1_overlap = text1[-int(overlap_ratio * len(text1)):]
            text2_overlap = text2[:int(overlap_ratio * len(text2))]
            split_index1 = min([text1_overlap.rfind(delimiter) for delimiter in delimiters])
            split_index2 = max([text2_overlap.find(delimiter) for delimiter in delimiters])
            page1.text += text2[:split_index2]
            page2.text = text1[split_index1:] + page2.text

    def split(self) -> list[DocumentChunk]:
        nodes = self.pipeline.run(
            documents=self.documents)
        return [DocumentChunk.from_llama_index_node(node) for node in nodes]


if __name__ == "__main__":
    ds = DocumentLoader()
    ds.load_directory("data")
    print(len(ds.documents))
    print(ds.documents[0].metadata)
    print(ds.documents[0])
    print(ds.split())
