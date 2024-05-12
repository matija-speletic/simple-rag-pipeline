from pathlib import Path

from rag import RAG
from llm import OllamaLLM, OllamaEmbeddingModel
from vector_store import Neo4jVectorStore

rag = RAG(
    Neo4jVectorStore(uri="bolt://localhost:7687", user="neo4j", password="neo4jneo4j"),
    OllamaEmbeddingModel(),
    OllamaLLM()
)

# rag.load_data_sources(
#     Path(r'C:\Users\matij\Projects\simple-rag-pipeline\data\KG-RAG-datasets\sec-10-q\data\v1\docs\aapl_only'),
#     reset_data_sources=True)

prompt = "What are the major factors contributing to the change in Apple's " \
         "gross margin in the most recent 10-Q compared to the previous quarters?"
response, context = rag.generate(prompt, [])
print(response)
# for r in response:
#     print(r)
for chunk in context:
    print("##############################################################################################")
    print(chunk.document_name)
    print(chunk.text)
