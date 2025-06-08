import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph

load_dotenv()
st.title("AI robot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

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

@st.cache_resource
def BuildVectorStore(filename: str):
    loader = PyPDFLoader(filename)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=5, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)

    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = InMemoryVectorStore(embedding_model)
    ids = vector_store.add_documents(documents = all_splits)
    return vector_store

#conversation = BuildConversation()
session_id = "user-123"
vector_store = BuildVectorStore(filename = 'data/test.pdf')

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state['question'])
    return {'context': retrieved_docs}

def generate(state: State):
    docs = '\n\n'.join(doc.page_content for doc in state['context'])
    prompt_msg = prompt.invoke({'question': state['question'], 'context': docs})
    model_resp = model.invoke(prompt_msg)
    return {'answer': model_resp.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, 'retrieve')
graph = graph_builder.compile()

if question := st.chat_input('ask anything'):
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        #raw_response = conversation.invoke({"input": question}, config={"configurable": {"session_id": session_id}})
        #st.write(raw_response.content)
        raw_response = graph.invoke({'question': question})
        response = raw_response['answer']
        st.write(response)

    st.session_state.messages.append({'role': 'assistant', 'content': response})