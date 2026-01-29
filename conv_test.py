from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

def main():
    llm = ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    messages = [
            SystemMessage(content=(
            "You are TrackSante's AI measurement coach. You are to ask concise questions "
            "to the user to get more information on their symptoms, health status, and lifestyle habits. "
            "Be empathetic, supportive, and non-judgmental in your responses. "
            "Keep your questions clear and focused on understanding the user's health needs."
        ))
    ]
    
    while(True):
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        messages.append(AIMessage(content=response.content))

        print("AI Measurement Coach:", response.content)

if __name__ == "__main__":
    main()