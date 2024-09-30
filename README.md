# Experimenting with RAG techniques in python

Working through https://medium.com/@suraj_bansal/build-your-own-ai-chatbot-a-beginners-guide-to-rag-and-langchain-0189a18ec401

# Usage

First set up as below, then:

`python stateless-bot.py`

# Setup

First download the paper "The Impact, Advancements and Applications of
Generative AI" and save it as `data/impact_of_generativeAI.pdf`

Install dependencies:

`pip install -r requirements.txt`

To bootstrap, export environment variables.
I'm using direnv for this:

```
export OPENAI_API_KEY="..."
export PINECONE_API_KEY="..."
export PINECONE_INDEX_NAME="pdf-vectorized"

```

Then run this (only needed once):

`python ingestion.py`

That should populate pinecone with your data for the Retrieval step.
