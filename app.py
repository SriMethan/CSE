import streamlit as st
import os
import tempfile
import hashlib
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 🔐 Secure API Key
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# 🚀 Page config
st.set_page_config(page_title="SriMethan AI • PDF Chat 🤖", layout="centered")

# 🧠 Session Init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# 🧠 Hashing for cache

def get_file_hash(files):
    md5 = hashlib.md5()
    for file in files:
        md5.update(file.getvalue())
    return md5.hexdigest()

# 📄 Main Upload Area
st.markdown("## 📄 Upload Your PDF(s)")
uploaded_files_top = st.file_uploader("Upload here to get started:", type=["pdf"], accept_multiple_files=True)

uploaded_files = uploaded_files_top

# 📚 Sidebar
with st.sidebar:
    st.markdown("### 🏢 **SriMethan Holdings (PVT) LTD**")
    st.markdown("Bringing your documents to life with AI ⚡")
    st.markdown("---")
    uploaded_files_sidebar = st.file_uploader("Re-upload your PDFs:", type=["pdf"], accept_multiple_files=True)
    if uploaded_files_sidebar:
        uploaded_files = uploaded_files_sidebar

# 📚 Process and Embed
if uploaded_files and not st.session_state.vectorstore_ready:
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

        with st.spinner("🔍 Processing and embedding your documents..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            vectorstore.save_local(db_path)
    else:
        with st.spinner("📂 Loading from previous session..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        streaming=True,
        temperature=0.2
    )

    qa_chain = load_qa_chain(llm, chain_type="stuff")

    st.session_state.qa_chain = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=qa_chain,
        return_source_documents=False
    )

    st.session_state.vectorstore_ready = True
    st.success("✅ Your files are ready. Start chatting below 👇")

# 📢 Title Header
st.markdown("# 🏢 SRIMETHAN HOLDINGS (PVT) LTD")

# 💬 Chat Interface
if st.session_state.vectorstore_ready:
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**You:** {q}")
        with st.chat_message("assistant"):
            st.markdown(f"**SriMethan Model 🤖:**\n\n{a}")

    query = st.chat_input("💬 Ask your next question...")
    if query:
        with st.chat_message("user"):
            st.markdown(f"**You:** {query}")

        response = ""
        with st.chat_message("assistant"):
            msg_box = st.empty()
            msg_box.markdown("**SriMethan Model 🤖:**\n\nThinking...")
            for chunk in st.session_state.qa_chain.stream({
                "question": query,
                "chat_history": st.session_state.chat_history
            }):
                token = chunk.get("answer", "")
                response += token
                msg_box.markdown(f"**SriMethan Model 🤖:**\n\n{response}")

        st.session_state.chat_history.append((query, response))

        # 📢 Footer
        st.markdown("""
        ---
        <div style='text-align: center; font-size: 0.9em;'>
        Powered by <strong>SriMethan Holdings (PVT) LTD</strong> • © 2025 All rights reserved.
        </div>
        """, unsafe_allow_html=True)
