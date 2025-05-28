from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


load_dotenv()


def BuildMsg(text):
    prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "Reply in Traditional Chinese."),
    ("user", "{text}"),
    ])
    return prompt.invoke({"text": text})

messages = BuildMsg("2乘3是多少？")



model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


filename = "data/nke-10k-2023.pdf"
loader = PyPDFLoader(filename)

print('Loading PDF...')
docs = loader.load()
print('Loading PDF Finished.')

print('Pages: {0}' .format(len(docs)))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

print('Splits: {0}' .format(len(all_splits)))

embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
vector1 = embedding_model.embed_query(all_splits[0].page_content)
vector2 = embedding_model.embed_query(all_splits[1].page_content)
print('length of vector1 and vector2 is {0} and {1}' .format(len(vector1), len(vector2)))

vector_store = InMemoryVectorStore(embedding_model)
ids = vector_store.add_documents(documents = all_splits)

results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")
print('size of result: {0}' .format(len(results)))
print(results[0])
results = vector_store.similarity_search_with_score("How many distribution centers does Nike have in the US?")
docs, score = results[0]
print('score: {0}\n{1}' .format(score, docs))
