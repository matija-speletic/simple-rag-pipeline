from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


DEFAULT_DISTRIBUTIONS = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}


def generate_testset(documents, output_path, test_size,
                     distributions=None):
    if distributions is None:
        distributions = DEFAULT_DISTRIBUTIONS
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    critic_llm = ChatOpenAI(model="gpt-4")
    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )

    testset = generator.generate_with_llamaindex_docs(
        documents, test_size=test_size, distributions=distributions)

    testset.to_pandas().to_csv(output_path)
