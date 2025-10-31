#OpenAI (GPT-4, GPT-4o, GPT-3.5-turbo)

from openai import OpenAI

client = OpenAI(api_key="your_openai_api_key")

response = client.chat.completions.create(
    model="gpt-4o-mini",  # or "gpt-4-turbo", "gpt-3.5-turbo"
    messages=[
        {"role": "system", "content": "You are a finance analyst."},
        {"role": "user", "content": "Explain RAG in AI."}
    ]
)

print(response.choices[0].message.content)

#Anthropic Claude 3
from anthropic import Anthropic

client = Anthropic(api_key="your_anthropic_api_key")

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=300,
    messages=[
        {"role": "user", "content": "Explain RAG with an example."}
    ]
)

print(response.content[0].text)

#Llama 3 (Local)
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="/models/llama-3-8b-instruct.Q4_K_M.gguf",
    n_ctx=2048,
    temperature=0.2
)

print(llm.invoke("Explain Retrieval Augmented Generation."))

#Mistral / Mixtral (Open Source)
from langchain_community.llms import Ollama

llm = Ollama(model="mixtral")
response = llm.invoke("What is the purpose of RAG?")
print(response)

#Google Gemini (via API)
import google.generativeai as genai

genai.configure(api_key="your_gemini_api_key")

model = genai.GenerativeModel("gemini-1.5-pro")
response = model.generate_content("Explain RAG in machine learning.")
print(response.text)
