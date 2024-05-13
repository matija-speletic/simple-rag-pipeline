from abc import ABC, abstractmethod
import os
from typing import Type

import llama_index.core.llms as li_llm
import llama_index.core.embeddings as li_emb
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI


CONTEXT_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the users question. 

{}

If you don't know the answer, just say that you "don't have information", only that and nothing more. 
Keep the answer concise and to the point."""

SYSTEM_PROMPT_TEMPLATE = """
If you don't know the answer, just say that you don't know, don't try to make up an answer."""


LLMS: dict[str, tuple[Type[li_llm.LLM], dict]] = {
    'phi3': (Ollama, {'model': 'phi3', 'request_timeout': 100}),
    'gpt-3.5-turbo': (OpenAI, {'api_key': os.environ.get('OPENAI_API_KEY'), 'model': 'gpt-3.5-turbo'}),
}

EMBEDDING_MODELS: dict[str, tuple[Type[li_emb.BaseEmbedding], dict]] = {
    'nomic-embed-text': (OllamaEmbedding, {'model_name': 'nomic-embed-text'}),
    'phi3': (OllamaEmbedding, {'model_name': 'phi3'}),
}


class LLM:
    def __init__(self, model: li_llm.LLM | str,
                 system_prompt: str | None = SYSTEM_PROMPT_TEMPLATE,
                 context_prompt: str | None = CONTEXT_PROMPT_TEMPLATE, ):
        if isinstance(model, str):
            model_class, model_kwargs = LLMS[model]
            self.model = model_class(**model_kwargs)
        else:
            self.model = model
        self.system_prompt = system_prompt
        self.context_prompt = context_prompt

    def generate(self, prompt: str,
                 history: list[tuple[li_llm.MessageRole, str]],
                 context: str | list[str] | None = None) -> str:
        _chat = self._prepare_chat(context, history, prompt)
        return self.model.chat(_chat).message.content

    def _prepare_chat(self, context: str | list[str],
                      history: list[tuple[li_llm.MessageRole, str]],
                      prompt: str):
        if self.context_prompt and context and len(context) > 0:
            if isinstance(context, list):
                context = "\n\n".join(context)
            _system_prompt = self.context_prompt.format(context)
        else:
            _system_prompt = None

        _chat = [ChatMessage(content=content, role=role) for role, content in history]
        if _system_prompt:
            _chat.append(ChatMessage(content=_system_prompt, role=MessageRole.SYSTEM))
        if len(_chat) == 0:
            _chat.append(ChatMessage(content=self.system_prompt, role=MessageRole.SYSTEM))
        _chat.append(ChatMessage(content=prompt, role=MessageRole.USER))
        return _chat


class Embedding:
    def __init__(self, model: li_emb.BaseEmbedding | str):
        if isinstance(model, str):
            model_class, model_kwargs = EMBEDDING_MODELS[model]
            self.model = model_class(**model_kwargs)
        else:
            self.model = model
        self.show_progress = False

    def get_query_embedding(self, query: str) -> list[float]:
        return self.model.get_query_embedding(query)

    def get_text_embedding_batch(self, text_list: list[str]) -> list[list[float]]:
        return self.model.get_text_embedding_batch(
            text_list, show_progress=self.show_progress)
