import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
import pandas as pd

from rag import RAG


def load_data_sources(rag: RAG, data_sources_dir: str):
    rag.load_data_sources(
        Path(data_sources_dir),
        reset_data_sources=True,
        chunk_size=1024,
        chunk_overlap=128,
    )


def add_metadata_to_output_json(json_path: str, metadata: dict):
    with open(json_path, 'r') as file:
        data: dict = json.load(file)
    data['metadata'] = metadata
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)


def append_to_output_file(file_path: str,
                          question: str, answer: str,
                          context: str, ground_truth: str):
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


def run_on_dataset(rag: RAG, qa_json_path: str, output_json_path: str, metadata=None):
    with open(qa_json_path) as f:
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
        append_to_output_file(output_json_path, q, response, _context, a)
    if metadata:
        add_metadata_to_output_json(output_json_path, metadata)


def run_query(rag: RAG, query: str):
    response, context = rag.generate(query, [])
    print('Question:')
    print(query, end='\n\n')

    print('Answer:')
    print(response, end='\n\n')

    print('Context:')
    for chunk in context:
        print(chunk.text + "\n\nSOURCE: " + chunk.document_name + f"({chunk.page})")
    print("##############################################################################################")


def evaluate_rag(rag_output_path: str, results_path: str,
                 truncate=False, truncate_size=4):
    with open(rag_output_path) as f:
        data = json.load(f)
    if 'contexts' not in data and 'context' in data:
        data['contexts'] = data['context']
        del data['context']
    if 'metadata' in data:
        del data['metadata']
    if truncate:
        data['question'] = data['question'][:truncate_size]
        data['answer'] = data['answer'][:truncate_size]
        data['contexts'] = data['contexts'][:truncate_size]
        data['ground_truth'] = data['ground_truth'][:truncate_size]
    dataset = Dataset.from_dict(data)
    results = ragas_evaluate(dataset,
                             metrics=[
                                 answer_relevancy,
                                 faithfulness,
                                 context_recall,
                                 context_precision,
                                 answer_correctness,
                                 answer_similarity
                             ], raise_exceptions=False)
    results_df = results.to_pandas()
    results_df.to_json(results_path + '.json', indent=4)



