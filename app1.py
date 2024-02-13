import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

api_key = "AIzaSyAN8qDBM-rPoyB5exg83K62HH-d2iBi948"
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key, temperature=0.1)

urls = [
    "https://www.bodybuilding.com/content/8-benefits-of-high-intensity-interval-training-hiit.html",
    "https://www.menshealth.com/fitness/a19537337/best-biceps-exercises/",
    "https://www.healthline.com/nutrition/27-health-and-nutrition-tips"
]


st.title("Gym Expert System")

# User input for a question
question_input = st.text_input("Enter your question:")

if st.button("Get Answer"):
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # create embeddings and save it to FAISS index
    embeddings = HuggingFaceEmbeddings()
    vectorstore_open = FAISS.from_documents(docs, embeddings)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer, try to provide as much text as possible from the "response" section in the source document context without making many changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_open.as_retriever(),
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    result = chain(question_input)

    st.write("Your Question:", question_input)
    st.write("Best Matching Answer:", result["result"])

    if result["result"] == "I don't know.":
        st.warning("Sorry, the answer could not be found in the provided context.")
    else:
        st.success("Answer found successfully!")

    for idx, source_doc in enumerate(result["source_documents"]):
        st.subheader(f"Source Document {idx + 1}:")
        st.text(source_doc)
