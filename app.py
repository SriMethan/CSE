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
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain

# 🔐 Load secure API key
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# 🚀 Page config
st.set_page_config(page_title="SriMethan AI • PDF Chat 🤖", layout="centered")

# ----------------------
# 🎨 ChatGPT-style CSS
# ----------------------
st.markdown(
    """
    <style>
    /* App background (dark, ChatGPT vibe) */
    .stApp { background: #0e0f13; color: #e5e7eb; }

    /* Centered content column like ChatGPT */
    .main > div { display: flex; justify-content: center; }
    .block-container { max-width: 880px !important; width: 100%; padding-top: 1.5rem; }

    /* Top brand header bar */
    .brand-bar {
        position: sticky; top: 0; z-index: 50;
        background: rgba(14,15,19,0.9); backdrop-filter: blur(6px);
        border-bottom: 1px solid #1f2430; padding: 10px 0 14px 0; margin-bottom: 10px;
    }
    .brand-title { font-weight: 700; letter-spacing: .4px; color: #fafafa; }
    .brand-sub { color: #9aa1ac; font-size: 0.92rem; }

    /* Upload card */
    .upload-card {
        border: 1px dashed #2b3240; border-radius: 14px; padding: 18px 16px;
        background: #12141a;
    }
    .upload-title { font-weight: 600; color: #eaecef; }
    .upload-hint { color: #9aa1ac; font-size: 0.9rem; }

    /* Chat bubbles */
    .chat-bubble {
        border: 1px solid #1f2430; border-radius: 14px; padding: 14px 16px;
        background: #11141a; color: #e5e7eb;
    }
    .chat-bubble.assistant { background: #0f1319; }
    .chat-bubble.user { background: #151924; }
    .chat-name { font-weight: 600; font-size: 0.95rem; color: #cdd3dc; margin-bottom: 6px; }

    /* Divider + footer */
    .soft-divider { border-color: #1f2430 !important; }
    .footer-note { text-align: center; font-size: 0.9rem; color: #a9b1bc; }

    /* Hide default Streamlit junk spacing */
    header, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# 🧠 Session init
# ----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# 🧹 New chat button (resets history but keeps vectorstore to avoid re-embed)
col_a, col_b = st.columns([1, 5])
with col_a:
    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.chat_history = []

# ----------------------
# 🔠 Prompts (unchanged logic)
# ----------------------
DOC_PROMPT = PromptTemplate.from_template("""
You are a financial assistant. Use ONLY the context to answer the question concisely.

If the user asks for a specific metric (e.g., Profit Before Tax, Total Equity, EPS), return ONLY that metric in this exact format:
<Metric Name>: <value>

No extra sentences unless specifically asked.

Context:
{context}

Question: {question}

Answer:
""")

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the conversation and a follow-up question, rewrite the follow-up into a standalone question that can be answered from the context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:
""")

# ----------------------
# 🧮 Helpers
# ----------------------
def get_file_hash(files):
    md5 = hashlib.md5()
    for file in files:
        md5.update(file.getvalue())
    return md5.hexdigest()

def bubble(name: str, text: str, role: str):
    css_class = "assistant" if role == "assistant" else "user"
    st.markdown(
        f"""
        <div class="chat-bubble {css_class}">
            <div class="chat-name">{name}</div>
            <div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------
# 🧾 Header + Upload
# ----------------------
st.markdown(
    """
    <div class="brand-bar">
      <div class="brand-title">SRIMETHAN HOLDINGS (PVT) LTD</div>
      <div class="brand-sub">Private PDF Intelligence • SriMethan Model</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main uploader (collapsed label to avoid warning / keep accessibility)
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown('<div class="upload-title">📄 Upload your PDF(s)</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-hint">Drag & drop or click to select. You can also re-upload from the sidebar later.</div>', unsafe_allow_html=True)
uploaded_files_top = st.file_uploader(
    label="Upload PDFs (main)",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar uploader (keeps your layout)
with st.sidebar:
    st.markdown("### 🏢 **SriMethan Holdings (PVT) LTD**")
    st.markdown("Bringing your documents to life with AI ⚡")
    st.markdown("---")
    uploaded_files_sidebar = st.file_uploader(
        label="Upload PDFs (sidebar)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="sidebar_upload",
    )

uploaded_files = uploaded_files_sidebar or uploaded_files_top

# ----------------------
# 🧠 Vector store build
# ----------------------
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

        with st.spinner("🔍 Processing & embedding your documents..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            vectorstore.save_local(db_path)
    else:
        with st.spinner("📂 Loading vector cache..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever()

    # Prevent any OpenAI default env hijack
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_BASE_URL", None)

    # LLM via OpenRouter (DeepSeek behind the scenes, branded as SriMethan)
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528:free",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        streaming=True,
        temperature=0.2,
        default_headers={
            "HTTP-Referer": "https://streamlit.io/",
            "X-Title": "SriMethan PDF Chat",
        },
    )

    # Explicit internal chains (no pydantic prompt var errors)
    combine_docs_chain = load_qa_chain(llm, chain_type="stuff", prompt=DOC_PROMPT)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

    st.session_state.qa_chain = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        return_source_documents=False,
    )

    st.session_state.vectorstore_ready = True
    st.success("✅ PDFs ready. Start chatting below.")

# ----------------------
# 💬 Chat section (ChatGPT-like)
# ----------------------
if st.session_state.vectorstore_ready:
    # Render history as bubbles
    for q, a in st.session_state.chat_history:
        bubble("You", q, "user")
        bubble("SriMethan Model 🤖", a, "assistant")

    # Chat input (Streamlit handles sticky bottom like ChatGPT)
    query = st.chat_input("Send a message...")
    if query:
        bubble("You", query, "user")

        # Stream the answer cleanly
        response = ""
        with st.chat_message("assistant"):
            thinking = st.empty()
            thinking.markdown("_Thinking…_")
            try:
                for chunk in st.session_state.qa_chain.stream({
                    "question": query,
                    "chat_history": st.session_state.chat_history
                }):
                    response += chunk.get("answer", "")
                    # Show partial answer as a bubble-like block
                    thinking.markdown(
                        f"""
                        <div class="chat-bubble assistant">
                            <div class="chat-name">SriMethan Model 🤖</div>
                            <div>{response}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            except Exception:
                thinking.markdown(
                    """
                    <div class="chat-bubble assistant">
                      <div class="chat-name">SriMethan Model 🤖</div>
                      <div>Sorry, I couldn't reach the model. Check your API key / usage.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                raise

        st.session_state.chat_history.append((query, response))

    # Footer like ChatGPT
    st.markdown('<hr class="soft-divider" />', unsafe_allow_html=True)
    st.markdown(
        "<div class='footer-note'>Powered by <strong>SriMethan Holdings (PVT) LTD</strong> • © 2025 All rights reserved.</div>",
        unsafe_allow_html=True,
    )
else:
    # Empty state helper text (ChatGPT style)
    st.markdown(
        """
        <div style="text-align:center; color:#a9b1bc; margin-top:28px;">
            Start by uploading one or more PDFs above. Your chat will appear here.<br/>
            Tip: You can re-upload from the sidebar any time.
        </div>
        """,
        unsafe_allow_html=True,
    )
