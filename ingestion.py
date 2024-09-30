import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

print("Step 2. Loading PDF")
path = os.path.join(os.path.dirname(__file__), "data/impact_of_generativeAI.pdf")
loader = PyPDFLoader(path)
document = loader.load()
print(f"loaded {path}")

print("Step 3. Splitting text into chunks")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(document)
print(f"created {len(texts)} chunks")

# for i, text in enumerate(texts):
#     print(f"chunk {i}\n==========================================\n")
#     print(text)

print("Step 4. Create embeddings and load into Pinecone")

embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
# PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("PINECONE_INDEX_NAME"))
