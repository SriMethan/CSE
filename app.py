import streamlit as st
import os
import tempfile
import hashlib
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 🚀 Custom Branding
st.set_page_config(page_title="SriMethan AI 📄🤖", layout="centered")

# 🏢 Sidebar Header with Branding
with st.sidebar:
    st.markdown("## 🏢 SriMethan Holdings (PVT) LTD")
    st.markdown("Welcome to your private AI PDF chatbot! 🔐")
    openrouter_api_key = st.text_input("🔑 Your Model Access Key", type="password")
    st.markdown("---")
    st.markdown("Upload PDFs here ↓")

# 📄 PDF Upload (Multi-PDF)
uploaded_files = st.sidebar.file_uploader("", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

# 🧠 Initialize Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# 📦 PDF Hash for Caching
def get_file_hash(files):
    md5 = hashlib.md5()
    for file in files:
        md5.update(file.getvalue())
    return md5.hexdigest()

# 🔄 Build or Load Vectorstore
if uploaded_files and openrouter_api_key and not st.session_state.vectorstore_ready:
    file_hash = get_file_hash(uploaded_files)
    db_path = f".cached_vectorstores/{file_hash}"

    if not os.path.exists(db_path):
        os.makedirs(".cached_vectorstores", exist_ok=True)
        all_docs = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                loader = PyPDFLoader(tmp_file.name)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                all_docs.extend(chunks)

        with st.spinner("🔍 Embedding your documents..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            vectorstore.save_local(db_path)
    else:
        with st.spinner("📂 Loading from cache..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever()

    # 🤖 Initialize Custom AI (Powered by DeepSeek but branded)
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=openrouter_api_key,
        streaming=True,
        temperature=0.2
    )

    # 🧠 Build the Conversational Chain
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    st.session_state.vectorstore_ready = True
    st.success("✅ PDF Ready! Ask anything 👇")

# 🧠 Chat Loop
if st.session_state.vectorstore_ready:
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**You:** {q}")
        with st.chat_message("assistant"):
            st.markdown(f"**SriMethan Model 🤖:** {a}")

    query = st.chat_input("💬 Ask your next question...")
    if query:
        with st.chat_message("user"):
            st.markdown(f"**You:** {query}")

        response = ""
        with st.chat_message("assistant"):
            msg_box = st.empty()
            for chunk in st.session_state.qa_chain.stream({
                "question": query,
                "chat_history": st.session_state.chat_history
            }):
                token = chunk.get("answer", "")
                response += token
                msg_box.markdown(f"**SriMethan Model 🤖:** {response}")

        st.session_state.chat_history.append((query, response))

# 🔚 Footer
st.markdown("---")
st.markdown("Made with 💙 by **SriMethan Holdings (PVT) LTD** • All rights reserved © 2025")
