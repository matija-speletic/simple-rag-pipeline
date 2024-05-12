from typing import Generator
from llm.base import LLM, EmbeddingModel, SYSTEM_PROMPT_TEMPLATE, CONTEXT_PROMPT_TEMPLATE

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


class OllamaLLM(LLM):
    def __init__(self, model: str = 'phi3',
                 system_prompt: str | None = SYSTEM_PROMPT_TEMPLATE,
                 context_prompt: str | None = CONTEXT_PROMPT_TEMPLATE, ):
        _model = Ollama(model=model, request_timeout=200)
        super().__init__(
            model=_model,
            system_prompt=system_prompt,
            context_prompt=context_prompt)

    def generate(self, prompt: str,
                 history: list[tuple[MessageRole, str]],
                 context: str | list[str] | None = None,
                 stream: bool = False, ) -> str:
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

        print(_chat, end='\n\n')

        # if stream:
        #     for chunk in self.model.stream_chat(_chat):
        #         yield chunk.message.content
        # else:
        return self.model.chat(_chat).message.content


class OllamaEmbeddingModel(EmbeddingModel):
    EMBEDDING_LENGTH: dict[str, int] = {
        'phi3': 3072,
        'nomic-embed-text': 768,
    }

    def __init__(self, model: str = 'nomic-embed-text'):
        self.model = OllamaEmbedding(model_name=model)
        self.show_progress = True

    def get_query_embedding(self, query: str) -> list[float]:
        return self.model.get_query_embedding(query)

    def get_text_embedding_batch(self, text_list: list[str]) -> list[list[float]]:
        return self.model.get_text_embedding_batch(
            text_list, show_progress=self.show_progress)
