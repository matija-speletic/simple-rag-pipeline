
from rag import RAG
from llm import LLM, Embedding
from vector_store import Neo4jVectorStore

from dotenv import load_dotenv
from rag.utils import (
    run_on_dataset,
    run_query,
    load_data_sources,
    evaluate_rag
)


load_dotenv(".env")

rag = RAG(
    Neo4jVectorStore(uri="bolt://localhost:7687", user="neo4j", password="neo4jneo4j",
                     embedding_size=768),
    Embedding('nomic-embed-text'),
    LLM('gpt-3.5-turbo')
)


# load_data_sources(r'med_qa/docs')
# run_query("What are the major factors contributing to the change in Apple's "
#           "gross margin in the most recent 10-Q compared to the previous quarters?")
# run_on_dataset(r'med_qa/qa.json', r'med_qa/output.json',
#                metadata={
#                     'dataset': 'med_qa',
#                     'vector_store': 'neo4j',
#                     'RAG': True,
#                     'chunk_size': 1024,
#                     'chunk_overlap': 128,
#                     'embedding': 'nomic-embed-text',
#                     'llm': 'gpt-3.5-turbo'})
# run_testset_generation(r'C:\Users\matij\Projects\opc-rag\pdf',
#                        r'C:\Users\matij\Projects\opc-rag\testset.csv', 10)

# prompt = "What are the major factors contributing to the change in Apple's " \
#          "gross margin in the most recent 10-Q compared to the previous quarters?"
# response, context = rag.generate(prompt, [])
# print(response)
# # for r in response:
# #     print(r)
# for chunk in context:
#     print("##############################################################################################")
#     print(chunk.document_name)
#     print(chunk.text)

# qa = pd.read_csv(r'data/KG-RAG-datasets/sec-10-q/data/v1/qna_data_aapl.csv')
# for i, row in qa.iterrows():
#     prompt = row['Question']
#     response, context = rag.generate(prompt, [])
#     print('Question:')
#     print(prompt, end='\n\n')
#
#     print('Answer:')
#     print(response, end='\n\n')
#
#     print('Ground Truth:')
#     print(row['Answer'], end='\n\n')
#     print("##############################################################################################")
# for chunk in context:
#     print(chunk.document_name)
#     print(chunk.text)
#     print("##############################################################################################")
