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

    
def BuildVectorStore(filename: str):
    loader = PyPDFLoader(filename)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=5, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = InMemoryVectorStore(embedding_model)
    ids = vector_store.add_documents(documents = all_splits)
    return vector_store


model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
prompt = hub.pull("rlm/rag-prompt")
vector_store = BuildVectorStore(filename = 'data/test.pdf')

    
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state['question'])
    return {'context': retrieved_docs}
    
def generate(state: State):
    docs = '\n\n'.join(doc.page_content for doc in state['context'])
    messages = prompt.invoke({'question': state['question'], 'context': docs})
    response = model.invoke(messages)
    return {'answer': response.content}
    

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, 'retrieve')
graph = graph_builder.compile()


result = graph.invoke({'question': 'i001是甚麼情況？'})
print('Answer: {0}' .format(result['answer']))

#with open("graph.png", "wb") as f:
#    f.write(graph.get_graph().draw_mermaid_png())