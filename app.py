import streamlit as st
import tempfile
import os

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="🧠 Chat with your PDF", layout="centered")
st.title("📄 Chat with your PDF 🤖 (Powered by DeepSeek on OpenRouter)")

openrouter_api_key = st.sidebar.text_input("🔐 Enter your OpenRouter API key", type="password")

uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_file and openrouter_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("📚 Reading your PDF..."):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)

    with st.spinner("🔎 Embedding + building vectorstore..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0.2
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("✅ Ready! Ask away 👇")

    query = st.text_input("❓ Ask your question")
    if query:
        with st.spinner("🤖 Thinking..."):
            response = qa_chain.run(query)
            st.markdown("### ✅ Answer:")
            st.write(response)

st.markdown("---")
st.markdown("Built with 💙 using Streamlit + LangChain + DeepSeek via OpenRouter")
