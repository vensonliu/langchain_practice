import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory


load_dotenv()
st.title("AI robot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

def BuildMsg(text):
    prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "Reply in Traditional Chinese."),
    ("user", "{text}"),
    ])
    return prompt.invoke({"text": text})

store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


@st.cache_resource
def BuildConversation():
    model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Reply in Traditional Chinese."),
        ("user", "{input}"),
    ])
    return RunnableWithMessageHistory(
        prompt | model,
        get_session_history,
        input_messages_key = "input"
    )

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


conversation = BuildConversation()

session_id = "user-123"

if prompt := st.chat_input('ask anything'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        raw_response = conversation.invoke({"input": prompt}, config={"configurable": {"session_id": session_id}})
        response = st.write(raw_response.content)

    st.session_state.messages.append({'role': 'assistant', 'content': response})