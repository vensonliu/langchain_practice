import os
from google import genai
from dotenv import load_dotenv
from google.genai import types
from PIL import Image

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

#response = client.models.generate_content(
#    model="gemini-2.0-flash",
#    config=types.GenerateContentConfig(
#        #system_instruction="You are a cat. Your name is Neko.",
#        system_instruction="please reply in traditional chinese",
#        max_output_tokens=500,
#        temperature=0.1
#    ),
#    contents="Explain how AI works in a few words"
#    #contents="Hi there"
#)


#image = Image.open('data/cats.jpg')
#response = client.models.generate_content(
#    model="gemini-2.0-flash",
#    config=types.GenerateContentConfig(
#        max_output_tokens=500,
#        temperature=0.1
#    ),
#    contents=[image, "Tell me what's in this picture"]
#)

#print(response.text)


response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="please reply in traditional chinese",
    ),
    contents="我有三隻貓"
)
print(response.text)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="please reply in traditional chinese",
    ),
    contents="我有幾隻貓？"
)
print(response.text)


chats = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="please reply in traditional chinese",
    )
)

response = chats.send_message("我有三隻貓")
print(response.text)
response = chats.send_message("我有幾隻貓？")
print(response.text)
