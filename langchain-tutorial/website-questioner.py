#!/Users/paul/src/langchain-experiments-2024/.direnv/python-3.12/bin/python

import argparse
import re
import os.path
import textwrap
import logging

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


logger = logging.getLogger(__name__)


def get_text_splits_from_url(url):
    logger.debug(f"Fetching and splitting contents of {url}")
    loader = WebBaseLoader(
        web_paths=(url,),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits


def make_vector_db(collection_name: str, split_function: callable):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=DATA_DIR,
    )
    # Only add the documents to the vectorstore if it's empty
    has_data = bool(vectorstore.get(limit=1, include=[])["ids"])
    if has_data:
        logging.debug(f"Loaded vector store for {collection_name}")
    else:
        logging.info(
            f"Adding initial documets to the vector store {collection_name} from split data"
        )
        splits = split_function()
        vectorstore.add_documents(splits)
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_collection_name(url: str):
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"\W+", "-", url)
    url = url.strip("-_")
    url = url[:63]
    # (3) otherwise contains only alphanumeric characters, underscores or hyphens (-),
    # (4) contains no two consecutive periods (..),
    # and (5) is not a valid IPv4 address
    logging.debug(f"Collection name: {url}")
    return url


def main(url: str, question: str):
    # Normalize the url into a collection name
    collection_name = get_collection_name(url)

    def splits_function():
        return get_text_splits_from_url(url)

    vectorstore = make_vector_db(collection_name, splits_function)
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")  # This gets prompt text from https://smith.langchain.com/hub/rlm/rag-prompt
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)


if __name__ == "__main__":

    # Parse arguments and get optional URL and question
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="https://lilianweng.github.io/posts/2023-06-23-agent/"
    )
    parser.add_argument("--question", default="What is Task Decomposition?")
    parser.add_argument("--verbose", action="store_true")
    parsed_args = parser.parse_args()
    url = parsed_args.url
    question = parsed_args.question
    if parsed_args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    result = main(url, question)
    print()
    print(textwrap.fill(f"Question: {question}"))
    print(textwrap.fill(f"answering from url {url}"))
    print()
    print(textwrap.fill(result, replace_whitespace=False))
