from pathlib import Path

from rag import RAG
from llm import LLM, Embedding
from vector_store import Neo4jVectorStore
import pandas as pd
import json
from tqdm import tqdm
from dotenv import load_dotenv
from rag.document_loader import DocumentLoader
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
from datasets import Dataset


load_dotenv(".env")

rag = RAG(
    Neo4jVectorStore(uri="bolt://localhost:7687", user="neo4j", password="neo4jneo4j",
                     embedding_size=768),
    Embedding('nomic-embed-text'),
    LLM('gpt-3.5-turbo')
)


def append_to_output_file(file_path, question, answer, context, ground_truth):
    try:
        with open(file_path, 'r') as file:
            data: dict[str, list[str]] = json.load(file)
    except FileNotFoundError:
        data = {}
    if 'question' not in data:
        data['question'] = []
    if 'answer' not in data:
        data['answer'] = []
    if 'context' not in data:
        data['context'] = []
    if 'ground_truth' not in data:
        data['ground_truth'] = []
    data['question'].append(question)
    data['answer'].append(answer)
    data['context'].append(context)
    data['ground_truth'].append(ground_truth)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def load_data_sources(path):
    rag.load_data_sources(
        Path(path),
        reset_data_sources=True,
        chunk_size=1024,
        chunk_overlap=128,
    )


# def run_testset_generation(path, output_path, test_size=10):
#     dl = DocumentLoader()
#     dl.load_directory(path)
#     generate_testset(dl.documents, output_path, 10)

def add_metadata_to_output_json(path, metadata):
    with open(path, 'r') as file:
        data: dict[str, list[str]] = json.load(file)
    data['metadata'] = metadata
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def run_on_dataset(path, output, metadata=None):
    with open(path) as f:
        qa: dict[str, list[str]] = json.load(f)
    for q, a in zip(qa['question'], qa['answer']):
        try:
            response, context = rag.generate(q, [])
        except Exception as e:
            print(f"Error: {e}")
            response, context = "Failed (took too long)", []

        print('Question:')
        print(q, end='\n\n')

        print('Answer:')
        print(response, end='\n\n')

        print('Ground Truth:')
        print(a, end='\n\n')
        _context = [chunk.text + "\n\nSOURCE: " + chunk.document_name
                    for chunk in context]
        print("##############################################################################################")
        append_to_output_file(output, q, response, _context, a)
    if metadata:
        add_metadata_to_output_json(output, metadata)


def run_query(query):
    response, context = rag.generate(query, [])
    print('Question:')
    print(query, end='\n\n')

    print('Answer:')
    print(response, end='\n\n')

    print('Context:')
    for chunk in context:
        print(chunk.text + "\n\nSOURCE: " + chunk.document_name + f"({chunk.page})")
    print("##############################################################################################")


def evaluate_rag(rag_output, results_path, turncate=False, truncate_size=4):
    with open(rag_output) as f:
        data = json.load(f)
    if 'contexts' not in data and 'context' in data:
        data['contexts'] = data['context']
        del data['context']
    # if 'ground_truths' not in data and 'ground_truth' in data:
    #     data['ground_truths'] = data['ground_truth']
    #     del data['ground_truth']
    #     for i, gt in enumerate(data['ground_truths']):
    #         data['ground_truths'][i] = [gt]
    if 'metadata' in data:
        del data['metadata']
    if turncate:
        data['question'] = data['question'][:truncate_size]
        data['answer'] = data['answer'][:truncate_size]
        data['contexts'] = data['contexts'][:truncate_size]
        data['ground_truth'] = data['ground_truth'][:truncate_size]
    dataset = Dataset.from_dict(data)
    results = ragas_evaluate(dataset,
                             metrics=[
                                    # answer_relevancy,
                                    # faithfulness,
                                    # context_recall,
                                    # context_precision,
                                    answer_correctness,
                                    answer_similarity
                             ], raise_exceptions=False)
    results_df=results.to_pandas()
    results_df.to_json(results_path+'.json', indent=4)


evaluate_rag(r'med_qa/output.json', r'med_qa/ragas_results_ans')
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
