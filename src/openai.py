from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

embeddings = OpenAIEmbeddings()
chat_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
chat_llm_4 = ChatOpenAI(model="gpt-4", temperature=0)
chat_llm_4_stream = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
