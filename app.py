import streamlit as st
import os
import tempfile
import hashlib
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.messages import HumanMessage

# 🔐 API Key
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# 🚀 Page config
st.set_page_config(page_title="SriMethan AI", layout="centered")

# 🎨 Light theme + ChatGPT-style, with right/left alignment (unchanged design)
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
    background-color: #DCF8C6;
    color: #111;
}
.chat-bubble.assistant {
    background-color: #F7F7F8;
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
    st.session_state.doc_text = ""          # concatenated PDF text fed directly to DeepSeek
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# 🧠 Helpers
def get_file_hash(files):
    md5 = hashlib.md5()
    for file in files:
        md5.update(file.getvalue())
    return md5.hexdigest()

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

_SMALLTALK = re.compile(r"^(ok(ay)?|thanks|thank you|cool|great|nice|hi|hello|hey|yo|bye|goodbye|lol|haha|😀|🙂|😉|👍|👌)\b", re.I)
def is_smalltalk(q: str) -> bool:
    return bool(_SMALLTALK.search(q.strip()))

# 📏 We’ll cap the document we pass to the model to avoid blowing the context.
# DeepSeek R1 on OpenRouter has a large window, but we’ll be safe.
MAX_DOC_CHARS = 120_000  # adjust if you want to push more

# 🧠 “Direct read” prompt (no retrieval, no embeddings, just raw doc text)
DIRECT_PROMPT = PromptTemplate.from_template("""
You are a helpful financial assistant for SriMethan Holdings (PVT) LTD.

Use ONLY the Document below to answer.
Answer style: ONE short, natural sentence. If the user asks a specific metric, include the value naturally (e.g., "Total assets are Rs 2.999T as at 31 Mar 2025.").
If the answer is not in the Document, say: "Not found in the provided document."

Document:
{doc}

User question: {question}

Answer:
""")

# Small talk / non-document prompt (also one short sentence)
GENERAL_CHAT_PROMPT = PromptTemplate.from_template("""
You are a friendly assistant for SriMethan Holdings (PVT) LTD.
Reply in ONE short sentence, natural and helpful.

User: {user_input}
Assistant:
""")

# 📄 Upload section (unchanged design)
st.markdown("<h1 style='text-align:center;'>👑 SRIMETHAN HOLDINGS (PVT) LTD</h1>", unsafe_allow_html=True)
st.markdown("## 📄 Upload Your PDF(s)")
uploaded_files_top = st.file_uploader(
    "Upload here to get started:", type=["pdf"], accept_multiple_files=True
)

# 📄 Sidebar (unchanged)
with st.sidebar:
    st.markdown("### 🏢 **SriMethan Holdings (PVT) LTD**")
    st.markdown("Bringing your documents to life with AI ⚡")
    st.markdown("---")
    uploaded_files_sidebar = st.file_uploader(
        "Re-upload your PDFs:", type=["pdf"], accept_multiple_files=True, key="sidebar_upload"
    )
    if uploaded_files_sidebar:
        uploaded_files_top = uploaded_files_sidebar

uploaded_files = uploaded_files_top

# 📚 Process PDFs (NO RAG): just read and concatenate text, store in session_state.doc_text
if uploaded_files and not st.session_state.vectorstore_ready:
    # Build concatenated text from all PDFs
    all_texts = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load()

        # Split into chunks to normalize whitespace then join; no embeddings
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        # Keep simple “section” separators to help the model parse
        joined = "\n\n".join([c.page_content.strip() for c in chunks if c.page_content.strip()])
        all_texts.append(joined)

    full_text = "\n\n====\n\n".join(all_texts).strip()
    # Truncate to safe size
    if len(full_text) > MAX_DOC_CHARS:
        full_text = full_text[:MAX_DOC_CHARS]

    st.session_state.doc_text = full_text
    st.session_state.vectorstore_ready = True
    st.success("✅ Your files are ready. Start chatting below 👇")

# 🤖 DeepSeek LLM (same config, no vectorstore)
if st.session_state.vectorstore_ready:
    # Prevent any OpenAI default env hijack
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

    # Prebuild chains for convenience (both 1-sentence outputs)
    direct_chain = LLMChain(llm=llm, prompt=DIRECT_PROMPT)
    general_chain = LLMChain(llm=llm, prompt=GENERAL_CHAT_PROMPT)

    # Render history
    for q, a in st.session_state.chat_history:
        render_user(q)
        render_assistant(a)

    # Input
    query = st.chat_input("💬 Type your next question...")
    if query:
        render_user(query)

        # If small talk / generic → answer one short sentence without the doc
        if is_smalltalk(query):
            resp = general_chain.invoke({"user_input": query}).get("text", "").strip()
            if not resp:
                resp = "Got it!"
            render_assistant(resp)
            st.session_state.chat_history.append((query, resp))
        else:
            # Stream the answer using the direct doc prompt
            response = ""
            placeholder = st.empty()
            placeholder.markdown(
                "<div class='chat-row assistant'><div class='chat-bubble assistant'><b>SriMethan Model :</b> Thinking... 🧠</div></div>",
                unsafe_allow_html=True,
            )

            # Compose a single message with the rendered prompt (no retrieval)
            rendered = DIRECT_PROMPT.format(doc=st.session_state.doc_text, question=query)
            try:
                # Stream chunks
                for chunk in llm.stream([HumanMessage(content=rendered)]):
                    piece = getattr(chunk, "content", "") or ""
                    response += piece
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

            response = response.strip() or "Not found in the provided document."
            st.session_state.chat_history.append((query, response))

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; font-size: 0.9em;'>"
        "Powered by <strong>SriMethan Holdings (PVT) LTD</strong> • © 2025 All rights reserved."
        "</div>",
        unsafe_allow_html=True,
    )
