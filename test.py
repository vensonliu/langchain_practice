import getpass
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


messages = [
    SystemMessage("Reply in traditional Chinese"),
    HumanMessage("Langchain具有哪些功能？"),
]

model.invoke(messages)