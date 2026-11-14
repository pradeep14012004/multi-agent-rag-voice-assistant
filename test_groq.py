from groq import Groq
from app.config import GROQ_API_KEY, GROQ_MODEL

print("Testing Groq connection...")

client = Groq(api_key=GROQ_API_KEY)

response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {
            "role": "user",
            "content": "Explain supervised learning in two sentences."
        }
    ],
    temperature=0.1,
    max_completion_tokens=200
)

print("Groq response:")
print(response.choices[0].message.content)