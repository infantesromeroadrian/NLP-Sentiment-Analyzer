import os
from langchain import OpenAI

# Pedir al usuario que ingrese la clave API de OpenAI
api_key = input("Por favor, ingrese su clave API de OpenAI: ")

# Configuración de la clave API como variable de entorno
os.environ["OPENAI_API_KEY"] = api_key


class LangChainTweetSummarizer:
    def __init__(self, api_key):
        self.llm = OpenAI(api_key=api_key)

    def summarize_tweets(self, tweets):
        combined_tweets = " ".join(tweets)
        prompt = f"Resumir los siguientes tweets: {combined_tweets}"
        summary = self.llm.complete(prompt=prompt, max_tokens=100)
        return summary