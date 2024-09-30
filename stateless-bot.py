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

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    prompts = (
        "What are the applications of generative AI according to the paper? Please number each application.",
        "Can you please summarize what the paper says about application number 2?"
        "Say more about that."
        )

    for prompt in prompts:
        res = qa({"question": prompt, "chat_history": chat_history})
        chat_history.append((res["question"], res["answer"]))
        print(f"Q: {res['question']}")
        print('-----------------------------------')
        print(f"A: {res['answer']}\n\n")

if __name__ == "__main__":
    main()
