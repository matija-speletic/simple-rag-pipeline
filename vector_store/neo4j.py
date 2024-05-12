import logging
from typing import Literal

from neo4j import GraphDatabase, ManagedTransaction
from neo4j.graph import Node

from document_chunk import Chunk
from vector_store.base import VectorStore


class Neo4jVectorStore(VectorStore):
    def __init__(self, uri, user, password,
                 index_name='embedding_index',
                 chunk_label='Chunk',
                 embedding_property='embedding',
                 embedding_size=3072,
                 similarity: Literal['cosine', 'euclidean'] = 'cosine',
                 document_label='Document',
                 chunk_relationship='BELONGS_TO_DOCUMENT'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.index_name = index_name
        self.chunk_label = chunk_label
        self.embedding_property = embedding_property
        self.embedding_size = embedding_size
        self.similarity = similarity
        self.document_label = document_label
        self.chunk_relationship = chunk_relationship
        self.create_index()

    def close(self):
        self.driver.close()

    def create_index(self):
        with self.driver.session() as session:
            session.write_transaction(self._create_index)

    def add_batch_chunks(self, chunks: list[Chunk]):
        with self.driver.session() as session:
            _doc_chunks: dict[str, list[Chunk]] = {}
            for _chunk in chunks:
                if _chunk.document_name not in _doc_chunks:
                    _doc_chunks[_chunk.document_name] = []
                _doc_chunks[_chunk.document_name].append(_chunk)
            for _document in _doc_chunks:
                _doc_path = _doc_chunks[_document][0].document_path
                _doc_node = session.write_transaction(self._add_document, _document, _doc_path)
                for _chunk in _doc_chunks[_document]:
                    session.write_transaction(self._add_chunk_to_index, _chunk, _document)

    def retrieve(self,
                 embedding: list[float],
                 nearest_neighbors: int = 5) -> list[tuple[Chunk, float]]:
        with self.driver.session() as session:
            _result: list[tuple[Node, Node, float]] = session.read_transaction(
                self._find_similar_nodes, embedding, nearest_neighbors)
            return [(Chunk(
                text=_node['text'],
                document_name=_doc['name'],
                page=_node['page'],
                embedding=_node[self.embedding_property]
            ), _score) for _node, _doc, _score in _result]

    def clear_data(self):
        def _delete_all(tx: ManagedTransaction):
            tx.run("MATCH (n) DETACH DELETE n")

        with self.driver.session() as session:
            session.write_transaction(_delete_all)

    def _create_index(self, tx: ManagedTransaction):
        query = (
            f"CREATE VECTOR INDEX `{self.index_name}` "
            "IF NOT EXISTS "
            f"FOR (n:{self.chunk_label}) "
            f"ON (n.{self.embedding_property}) "
            "OPTIONS {indexConfig: {"
            f"  `vector.dimensions`: {self.embedding_size}, "
            f"  `vector.similarity_function`: '{self.similarity}' "
            "}}"
        )
        logging.info(f"Creating index: {query}")
        try:
            tx.run(query)
        except Exception as e:
            logging.error(f"Error creating index: {e}")

    def _add_chunk_to_index(self, tx, chunk: Chunk, document_name: str):
        query = (
            f"MATCH (d:{self.document_label} {{name: $document_name}}) "
            f"CREATE (n:{self.chunk_label} {{"
            "text: $text, page: $page})"
            f" SET n.{self.embedding_property} = $embedding "
            f"CREATE (n)-[:{self.chunk_relationship}]->(d)"
            "RETURN n"
        )
        logging.info(f"Adding node to index: {query}")
        return tx.run(
            query,
            text=chunk.text,
            page=chunk.page,
            embedding=chunk.embedding,
            document_name=document_name
        ).single()[0]

    def _add_document(self, tx, document_name: str, document_path: str):
        query = (
            f"MERGE (n:{self.document_label} {{"
            "name: $name, link: $link}) "
            "ON CREATE SET n.date = datetime() "
            "RETURN n"
        )
        logging.info(f"Adding document to index: {query}")
        return tx.run(
            query, name=document_name, link=document_path).single()[0]

    def _find_similar_nodes(self, tx,
                            embedding: list[float],
                            nearest_neighbors: int = 5):
        query = (
            f"CALL db.index.vector.queryNodes($index, $k, $vector) "
            f"YIELD node, score "
            f"MATCH (node)-[:{self.chunk_relationship}]->(doc) "
            "RETURN node, doc, score "
            "ORDER BY score DESC"
        )
        return tx.run(query, index=self.index_name, k=nearest_neighbors, vector=embedding).values()


# Usage example:
# uri = "bolt://localhost:7687"
# user = "neo4j"
# password = "password"
# index_name = "my_vector_index"
# vector_index = VectorIndex(uri, user, password, index_name)
# vector_index.add_node_to_index(node_id=1, vector=[1, 2, 3])
# similar_nodes = vector_index.find_similar_nodes(vector=[1, 2, 3])
# for node, score in similar_nodes:
#     print(node, score)
# vector_index.close()


if __name__ == '__main__':
    # uri = "bolt://localhost:7687"
    # user = "neo4j"
    # password = "neo4jneo4j"
    vector_index = VectorIndex(uri="bolt://localhost:7687", user="neo4j", password="neo4jneo4j")
