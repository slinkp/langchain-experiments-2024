import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
import warnings

warnings.filterwarnings("ignore")

chat_history = []

def main():
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

    prompts = (
        "What are the applications of generative AI according to the paper? Please number each application.",
        "Can you please elaborate more on application number 2?"
        )

    for prompt in prompts:
        res = qa.invoke(prompt)
        print(f"Q: {res['query']}")
        print('-----------------------------------')
        print(f"A: {res['result']}\n\n")

if __name__ == "__main__":
    main()
