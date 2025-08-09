import streamlit as st
import os
import tempfile
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# 🔐 API Key
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# 🚀 Page config
st.set_page_config(page_title="SriMethan AI", layout="centered")

# 🎨 Light theme + ChatGPT-style, with right/left alignment
st.markdown("""
<style>
body {
    background-color: #ffffff;
    font-family: "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
}
/* Message row containers */
.chat-row {
    display: flex;
    margin: 8px 0;
}
.chat-row.user { justify-content: flex-end; }
.chat-row.assistant { justify-content: flex-start; }
/* Bubbles */
.chat-bubble {
    padding: 10px 14px;
    border-radius: 12px;
    max-width: 80%;
    box-shadow: 0px 1px 2px rgba(0,0,0,0.08);
    border: 1px solid #eee;
    line-height: 1.45;
    word-wrap: break-word;
    white-space: pre-wrap;
}
.chat-bubble.user {
    background-color: #DCF8C6; /* light green */
    color: #111;
}
chat-bubble.assistant {}
.chat-bubble.assistant {
    background-color: #F7F7F8; /* subtle gray */
    color: #111;
}
/* Hide default Streamlit header/footer chrome */
footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 🧠 Session init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""          # concatenated PDF text (optional)
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# 🧠 Helpers
def render_user(text: str):
    st.markdown(
        f"<div class='chat-row user'><div class='chat-bubble user'><b>You:</b> {text}</div></div>",
        unsafe_allow_html=True,
    )

def render_assistant(text: str):
    st.markdown(
        f"<div class='chat-row assistant'><div class='chat-bubble assistant'><b>SriMethan Model :</b> {text}</div></div>",
        unsafe_allow_html=True,
    )

def get_file_hash(files):
    md5 = hashlib.md5()
    for file in files:
        md5.update(file.getvalue())
    return md5.hexdigest()

# 📏 Soft cap on doc text to keep latency sane (DeepSeek has big context though)
MAX_DOC_CHARS = 160_000

# 🔥 Single universal prompt (NO routing, NO RAG)
# - Always answer in ≤ 1–2 short sentences (aim 25 words max).
# - You MAY use the Document if it helps; otherwise ignore it and answer normally.
DIRECT_PROMPT = PromptTemplate.from_template("""
You are SriMethan's assistant. Answer anything.
Keep replies extremely brief: ideally one short sentence (max ~25 words).
If the Document below is relevant, use it; otherwise ignore it and answer from general knowledge.
If you truly lack info, say so briefly (no long explanations).

Document (optional):
{doc}

User: {question}
Assistant:
""")

# 📄 Upload section
st.markdown("<h1 style='text-align:center;'>👑 SRIMETHAN HOLDINGS (PVT) LTD</h1>", unsafe_allow_html=True)
st.markdown("## 📄 Upload Your PDF(s)")
uploaded_files_top = st.file_uploader("Upload here to get started:", type=["pdf"], accept_multiple_files=True)

# 📄 Sidebar (brand)
with st.sidebar:
    st.markdown("### 🏢 **SriMethan Holdings (PVT) LTD**")
    st.markdown("Bringing your documents to life with AI ⚡")
    st.markdown("---")
    uploaded_files_sidebar = st.file_uploader("Re-upload your PDFs:", type=["pdf"], accept_multiple_files=True, key="sidebar_upload")
    if uploaded_files_sidebar:
        uploaded_files_top = uploaded_files_sidebar

uploaded_files = uploaded_files_top

# 📚 Read PDFs (NO embeddings, NO retriever): just extract + concat text
if uploaded_files and not st.session_state.vectorstore_ready:
    all_texts = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        joined = "\n\n".join([c.page_content.strip() for c in chunks if c.page_content.strip()])
        all_texts.append(joined)

    full_text = "\n\n====\n\n".join(all_texts).strip()
    if len(full_text) > MAX_DOC_CHARS:
        full_text = full_text[:MAX_DOC_CHARS]

    st.session_state.doc_text = full_text
    st.session_state.vectorstore_ready = True
    st.success("✅ Your files are ready. Start chatting below 👇")

# 🤖 DeepSeek (via OpenRouter), always used
if st.session_state.vectorstore_ready:
    # Kill any OpenAI defaults that might hijack
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_BASE_URL", None)

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

    # Render history
    for q, a in st.session_state.chat_history:
        render_user(q)
        render_assistant(a)

    # Input (single universal flow)
    query = st.chat_input("💬 Type your next question...")
    if query:
        render_user(query)

        response = ""
        placeholder = st.empty()
        placeholder.markdown(
            "<div class='chat-row assistant'><div class='chat-bubble assistant'><b>SriMethan Model :</b> Thinking... 🧠</div></div>",
            unsafe_allow_html=True,
        )

        # Build the one-shot prompt with doc attached (optional)
        rendered = DIRECT_PROMPT.format(
            doc=st.session_state.doc_text or "(no document provided)",
            question=query
        )

        try:
            # Stream directly (no retrieval, no routing)
            for chunk in llm.stream([HumanMessage(content=rendered)]):
                piece = getattr(chunk, "content", "") or ""
                response += piece
                # keep it snappy in UI as it streams
                placeholder.markdown(
                    f"<div class='chat-row assistant'><div class='chat-bubble assistant'><b>SriMethan Model :</b> {response}</div></div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            placeholder.markdown(
                "<div class='chat-row assistant'><div class='chat-bubble assistant'><b>SriMethan Model :</b> ❌ Error connecting to model.</div></div>",
                unsafe_allow_html=True,
            )
            raise

        # Final tidy (trim whitespace; optional hard cap for safety)
        response = (response or "").strip()
        st.session_state.chat_history.append((query, response or "I don't have enough info to answer."))
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; font-size: 0.9em;'>"
        "Powered by <strong>SriMethan Holdings (PVT) LTD</strong> • © 2025 All rights reserved."
        "</div>",
        unsafe_allow_html=True,
    )
