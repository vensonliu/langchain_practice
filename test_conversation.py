from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory



load_dotenv()


store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

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

conversation = BuildConversation()

session_id = "user-123"

response = conversation.invoke({"input": "我有三隻貓"}, config={"configurable": {"session_id": session_id}})
print(response.content)
response = conversation.invoke({"input": "我有幾隻貓？"}, config={"configurable": {"session_id": session_id}})
print(response.content)