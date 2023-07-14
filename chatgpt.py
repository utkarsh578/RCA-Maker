import os
import streamlit as st

import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma


import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

PERSIST = False

loader = TextLoader("data/data.txt")  # Use this line if you only need data.txt
loader.load()

if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"), chain_type="stuff",
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

st.title("Utkarsh's GPT Test")
prompt = st.text_input("Ask your question")


if prompt:
    st.write(index.query(prompt, llm=ChatOpenAI()))
