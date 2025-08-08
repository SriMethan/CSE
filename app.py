import streamlit as st
import tempfile
import os

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 🔧 Streamlit Page Setup
st.set_page_config(page_title="🧠 Chat with your PDF", layout="centered")
st.title("📄 Chat with your PDF 🤖 (Powered by DeepSeek on OpenRouter)")

# 🔐 API Key Input
openrouter_api_key = st.sidebar.text_input("🔐 Enter your OpenRouter API key", type="password")

# 📄 Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_file and openrouter_api_key:
    # 🔽 Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # 📚 Load and chunk PDF
    with st.spinner("📚 Reading your PDF..."):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)

    # 🧠 Embed and store vectors
    with st.spinner("🔎 Embedding + building vectorstore..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

    # 🤖 DeepSeek via OpenRouter — CORRECT MODEL!
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528:free",  # ✅ the correct model ID
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=openrouter_api_key,       # ✅ correct param name
        temperature=0.2
    )

    # 🔗 RAG chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("✅ Your PDF is ready! Ask anything 👇")

    # 💬 User query input
    query = st.text_input("❓ Ask your question about the PDF:")
    if query:
        with st.spinner("🤖 DeepSeek is thinking..."):
            response = qa_chain.run(query)
            st.markdown("### ✅ Answer:")
            st.write(response)

# Footer
st.markdown("---")
st.markdown("Built with 💙 using Streamlit + LangChain + DeepSeek via OpenRouter")
