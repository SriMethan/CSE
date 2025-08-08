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

# Set up Streamlit page
st.set_page_config(page_title="🧠 Chat with your PDFs", layout="centered")
st.title("📄 Multi-PDF Chatbot 🤖 (DeepSeek + Memory + Streaming + Caching)")

# API key
openrouter_api_key = st.sidebar.text_input("🔐 Enter your OpenRouter API key", type="password")

# Upload multiple PDFs
uploaded_files = st.file_uploader("📄 Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# Memory store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# Helper: create hash for PDF cache
def get_file_hash(files):
    md5 = hashlib.md5()
    for file in files:
        md5.update(file.getvalue())
    return md5.hexdigest()

# Handle vectorstore caching + loading
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

        with st.spinner("🔍 Embedding documents..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            vectorstore.save_local(db_path)
    else:
        with st.spinner("📂 Loading cached vectorstore..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever()

    # Initialize DeepSeek LLM with streaming
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=openrouter_api_key,
        streaming=True,
        temperature=0.2
    )

    # Memory + streaming QA chain
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    st.session_state.vectorstore_ready = True
    st.success("✅ Your PDFs are ready! You can chat below 👇")

# Display full chat history
if st.session_state.vectorstore_ready:
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(f"**You:** {q}")
        with st.chat_message("assistant"):
            st.markdown(f"**DeepSeek:** {a}")

    # Multi-question input loop with memory
    query = st.chat_input("💬 Ask your question...")
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
                msg_box.markdown(f"**DeepSeek:** {response}")

        st.session_state.chat_history.append((query, response))

st.markdown("---")
st.markdown("Built with 💙 using Streamlit + LangChain + DeepSeek via OpenRouter")
