import os
import textwrap
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []


def main():
    print("Step 1. Loading knowledge")
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    print("Step 2, building RAG chain")
    llm = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Below we use create_stuff_documents_chain to feed all retrieved context
    # into the LLM.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    # TODO finish setting up chain with create_retrieval_chain
    # as per https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html

    print("Step 3. Asking questions and getting answers")
    prompts = (
        "What are the applications of generative AI according to the paper? Please number each application.",
        "Can you please summarize what the paper says about application number 2 in the previous response?",
        "Say more about that. At least 5 sentences please.",
        "Can you rephrase that in 4 sentences or less for a first-grade reading level?",
    )

    for prompt in prompts:
        res = rag_chain.invoke({"input": prompt, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=res["input"]))
        chat_history.append(AIMessage(content=res["answer"]))
        print(f"Q: {textwrap.fill(res['input'])}")
        print("---------------------------------------------------")
        print(f"A:\n{textwrap.fill(res['answer'])}\n\n")


if __name__ == "__main__":
    main()
