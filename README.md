# Experimenting with RAG techniques in python

A place for me to play with [Retrieval Augmented Generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)

All dependencies can be installed via `pip install -r requirements.txt`

API keys and so forth mentioned below should be set as environment variables.
I prefer to do this via [direnv](https://direnv.net/) so I have all the
`export` shell commands below saved in an `.envrc` file that I don't put in
source control.


## Medium tutorial "Build Your Own AI Chatbot: A Beginner’s Guide to RAG and LangChain"

This is in the `medium-tutorial` directory.

My first attempt was working through https://medium.com/@suraj_bansal/build-your-own-ai-chatbot-a-beginners-guide-to-rag-and-langchain-0189a18ec401

### Setup

```shell

pip install -r requirements.txt  # Using venv or similar recommended

cd medium-tutorial

# This file may have moved! Search for:
# "The Impact, Advancements and Applications of Generative AI"
# and download the PDF version here.
URL="https://www.internationaljournalssrg.org/IJCSE/2023/Volume10-Issue6/IJCSE-V10I6P101.pdf"
curl "$URL" > data/impact_of_generativeAI.pdf

# Set environment variables. Automate using direnv or similar if you like:

export OPENAI_API_KEY="..."
export PINECONE_API_KEY="..."
export PINECONE_INDEX_NAME="pdf-vectorized"

```

Then follow the tutorial to set up your pinecone data.
Once that's done, run this (only needed once):

```shell
python ingestion.py
```

That should populate pinecone with your data for the Retrieval step.

Now the bot should work!

### Running the bot

```console
$ python stateful-bot.py
Step 1. Loading knowledge
Step 2, building RAG chain
Step 3. Asking questions and getting answers
Q: What are the applications of generative AI according to the paper?
Please number each application.
---------------------------------------------------
A:
The applications of generative AI mentioned in the paper include: 1.
Generating realistic images, 2. Synthesizing new music compositions,
3. Creating lifelike characters in video games, and 4. Assisting in
drug discovery by designing novel molecules.


Q: Can you please summarize what the paper says about application number
2 in the previous response?
---------------------------------------------------
A:
Application number 2 mentioned in the paper refers to generative AI's
ability to synthesize new music compositions. Generative AI can
autonomously create music that closely resembles human-generated
compositions by leveraging deep learning techniques and generative
models. This application showcases how generative AI can be used to
produce creative content in the form of music.


Q: Say more about that. At least 5 sentences please.
---------------------------------------------------
A:
Generative AI's application in synthesizing new music compositions
involves using algorithms to analyze existing music data and generate
new pieces. By training on vast amounts of musical data, generative AI
models can learn patterns, styles, and structures to create original
compositions. These models can mimic different genres, artists, or
even blend styles to produce unique music. The generated music can be
used for various purposes, such as background scores for films, video
games, or even personalized music recommendations. Overall, the
ability of generative AI to create music opens up new possibilities
for artists, composers, and the entertainment industry as a whole.


Q: Can you rephrase that in 4 sentences or less for a first-grade reading
level?
---------------------------------------------------
A:
Generative AI helps make new music by learning from other songs. It
can create different kinds of music like in movies or games. The music
it makes is special and can sound like music made by people. This
helps artists and composers make more music for everyone to enjoy.
```

## Langchain official tutorial

Next up is this one: https://python.langchain.com/docs/tutorials/rag/

I did the Anthropic version. We need to set some more env vars; sign up for
langsmith and anthropic API access and:

```sh
export LANGCHAIN_TRACING_V2="true"
export LANGSMITH_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

Again, I do this via direnv, so I put those in an `.envrc` file instead of
using `getpass`.

That done, it Just Works:

```console
$ python langchain-tutorial/blog-rag-demo.py
USER_AGENT environment variable not set, consider setting it to identify your requests.

Task Decomposition is a technique for breaking down complex tasks into
smaller, more manageable steps. It is often used with large language
models to improve performance on challenging problems. Common
approaches include Chain of Thought prompting, Tree of Thoughts
searching, and using prompts or task-specific instructions to guide
the decomposition process.
```

### Enhancements

- Persisted the Chroma data store to disk
  - Only hit the URL if we don't have data (ie, first run)
