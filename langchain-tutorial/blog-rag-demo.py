import os.path
import bs4
import textwrap
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_anthropic import ChatAnthropic

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

def get_text_splits_from_url(url):
    print(f"Fetching and splitting contents of {url}")
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits


def make_vector_db():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma(
        collection_name="blog-vector-data",
        embedding_function=embeddings,
        persist_directory=DATA_DIR,
    )
    # Only add the documents to the vectorstore if it's empty
    has_data = bool(vectorstore.get(limit=1, include=[])["ids"])
    if has_data:
        print("Loaded vector store from disk")
    else:
        print("Adding documents to the vector store from split data")
        splits = get_text_splits_from_url(
            "https://lilianweng.github.io/posts/2023-06-23-agent/"
        )
        vectorstore.add_documents(splits)
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main(question: str):
    vectorstore = make_vector_db()
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")  # What's this mean?
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)


if __name__ == "__main__":
    result = main("What is Task Decomposition?")
    print(textwrap.fill(result))
