
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY missing in .env")

print("Connecting to Groq Cloud...")

client = Groq(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "What is RAG? Explain in one sentence."
            }
        ],
        temperature=0,
        max_completion_tokens=300
    )

    answer = response.choices[0].message.content

    print("\nGroq API connected successfully!")
    print("Model:", response.model)
    print("AI Response:", answer)

except Exception as e:
    print("\nGroq API Error:")
    print(type(e).__name__, str(e))
