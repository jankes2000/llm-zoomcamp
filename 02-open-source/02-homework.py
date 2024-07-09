from openai import OpenAI
import tiktoken

client = OpenAI(
    base_url= "http://localhost:11434/v1/",
    api_key="private",
)


prompt = "What's the formula for energy?"
response = client.chat.completions.create(
    model="gemma:2b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0
)


completion_text = response.choices[0].message.content
print(completion_text)

