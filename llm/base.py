from typing import Generator
from abc import ABC, abstractmethod

import llama_index.core.llms as li_llm

CONTEXT_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the users question. 

{}

If you don't know the answer, just say that you "don't have information", only that and nothing more. 
Keep the answer concise and to the point."""

SYSTEM_PROMPT_TEMPLATE = """
If you don't know the answer, just say that you don't know, don't try to make up an answer."""


class LLM(ABC):
    def __init__(self, model: li_llm.LLM,
                 system_prompt: str | None = SYSTEM_PROMPT_TEMPLATE,
                 context_prompt: str | None = CONTEXT_PROMPT_TEMPLATE, ):
        self.model = model
        self.system_prompt = system_prompt
        self.context_prompt = context_prompt

    @abstractmethod
    def generate(self, prompt: str,
                 history: list[tuple[li_llm.MessageRole, str]],
                 context: str | list[str] | None = None,
                 stream: bool = False, ) -> str:
        pass


class EmbeddingModel(ABC):
    @abstractmethod
    def get_query_embedding(self, query: str) -> list[float]:
        pass

    @abstractmethod
    def get_text_embedding_batch(self, text_list: list[str]) -> list[list[float]]:
        pass
