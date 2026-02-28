import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY is not set in your .env file.")

client = Mistral(api_key=MISTRAL_API_KEY)

print("Sending test request to Mistral API...")
response = client.chat.complete(
    model="mistral-small-2506",
    messages=[{"role": "user", "content": "Reply with 'API is working.' and nothing else."}]
)

print("Response:", response.choices[0].message.content)
