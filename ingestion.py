import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore


path = os.path.join(os.path.dirname(__file__), "data/impact_of_generativeAI.pdf")
loader = PyPDFLoader(path)
document = loader.load()
print(f"loaded {path}")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(document)
print(f"created {len(texts)} chunks")

# for i, text in enumerate(texts):
#     print(f"chunk {i}\n==========================================\n")
#     print(text)


# chat = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")
