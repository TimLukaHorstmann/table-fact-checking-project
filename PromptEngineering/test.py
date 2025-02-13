from langchain_ollama import OllamaLLM

model = OllamaLLM(model="mistral:latest")
model.invoke("what is the meaning of life?")