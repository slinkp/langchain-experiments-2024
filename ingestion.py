import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore


loader = PyPDFLoader("data/impact_of_generativeAI.pdf")
document = loader.load()

# chat = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")
