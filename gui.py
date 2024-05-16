import streamlit as st
from llama_index.core.llms import MessageRole

from rag import RAG
from llm import LLM, Embedding
from vector_store import Neo4jVectorStore

from dotenv import load_dotenv


load_dotenv(".env")

rag = RAG(
    Neo4jVectorStore(uri="bolt://localhost:7687", user="neo4j", password="neo4jneo4j",
                     embedding_size=768),
    Embedding('nomic-embed-text'),
    LLM('gpt-3.5-turbo', language='serbian')
)

st.title("Medical Trial QA Chatbot")

# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["messages"]
                ]

        def _get_message_role(role: str) -> MessageRole:
            if role == "assistant":
                return MessageRole.ASSISTANT
            return MessageRole.USER

        _history=[
            (_get_message_role(message["role"]), message["content"])
            for message in messages
        ]

        stream, chunks = rag.generate_stream(prompt, _history)
        response = st.write_stream(stream)
        _sources_html = "<details><summary><b>Sources</b></summary>\n<ol>\n"
        for chunk in chunks:
            _sources_html += f'<li><a href="{chunk.document_path}">{chunk.document_name}, page {chunk.page}</a></li><br>\n'
        _sources_html += "</ol>\n</details>"
        st.markdown(_sources_html, unsafe_allow_html=True)
        # st.markdown("<details><summary><b>Sources</b></summary>", unsafe_allow_html=True)
        # st.markdown("*SOURCES:*")
        # for chunk in chunks:
        #     st.markdown(f'- [{chunk.document_name}, page {chunk.page}]({chunk.document_path})',
        #                 unsafe_allow_html=True)
        # st.markdown("</details>", unsafe_allow_html=True)
    st.session_state["messages"].append({"role": "assistant", "content": response})
