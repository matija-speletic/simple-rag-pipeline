from document_splitter import DocumentSplitter
from vector_store import Neo4jVectorStore
from llm import OllamaEmbeddingModel
import logging


# logging.basicConfig(level=logging.INFO)
ds = DocumentSplitter()
# vs = Neo4jVectorStore(uri="bolt://localhost:7687", user="neo4j", password="neo4jneo4j")
# em = OllamaEmbeddingModel()
# ds.load_file(r"C:\Users\matij\Projects\simple-rag-pipeline\data\1703162885-ua-part-7-profiles-1.05.02-2022-11-01.pdf")
# chunks = ds.split()
# embeddings = em.get_text_embedding_batch([chunk.text for chunk in chunks])
# for chunk, embedding in zip(chunks, embeddings):
#     chunk.embedding = embedding
# vs.add_batch_to_index(chunks)
# qe = em.get_query_embedding("With regard to what are OPC UA profiles used for?")
# similar_nodes = vs.retrieve(qe)
# for chunk, score in similar_nodes:
#     print(chunk.text)
#     print(chunk.document_name)
#     print("##############################################################################################")

# vs.close()

ds.load_directory('data')
print(ds.documents)
