
import os
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "What is 2 + 2? Answer briefly."
            }
        ],
        max_completion_tokens=300
    )

    print("Answer:", repr(response.choices[0].message.content))
    print("Finish reason:", response.choices[0].finish_reason)

except RateLimitError as error:
    print("GROQ RATE LIMIT REACHED")
    print(error)

except Exception as error:
    print("API ERROR:", type(error).__name__, str(error))
