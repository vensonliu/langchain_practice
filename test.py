import getpass
import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


load_dotenv()

class ResponseFormatter(BaseModel):
    answer: str = Field(description = "Answer")
    followup_question: str = Field(description = "follow-up questions")


def BuildMsg(text):
    prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "Reply in Traditional Chinese."),
    ("user", "{text}"),
    ])
    return prompt.invoke({"text": text})



messages = BuildMsg("What is the powerhouse of the cell?")
#messages = BuildMsg("Langchain具有哪些功能？")

''' Usage 1 '''
#model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
#response = model.invoke(messages)
#print(response)

''' Usage 2 '''
#model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
#model_with_tools = model.bind_tools([ResponseFormatter])
#response = model_with_tools.invoke(messages)
#print(response)
#print()
#pydantic_obj = ResponseFormatter.model_validate(response.tool_calls[0]['args'])
#print(pydantic_obj)

''' Usage 3 '''
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai").with_structured_output(ResponseFormatter, method='json_mode')
response = model.invoke(messages)
print(response)