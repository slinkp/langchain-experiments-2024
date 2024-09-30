import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore


print("Step 1. Loading knowledge")
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
)

print("Step 2, building RAG chain and asking questions")
chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

qa = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
)
