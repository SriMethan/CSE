import streamlit as st
import os
import tempfile
import hashlib
import re
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain

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
    background-color: #DCF8C6; /* light green (iMessage style) */
    color: #111;
}
.chat-bubble.assistant {
    background-color: #F7F7F8; /* subtle gray like ChatGPT light */
    color: #111;
}

/* Hide default Streamlit header/footer chrome */
footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 🧠 Session init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "general_chain" not in st.session_state:
    st.session_state.general_chain = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# 🧠 File hash
def get_file_hash(files):
    md5 = hashlib.md5()
    for file in files:
        md5.update(file.getvalue())
    return md5.hexdigest()

# 🧠 Small-talk detector (to answer normally without RAG)
_SMALLTALK = re.compile(r"^(ok(ay)?|thanks|thank you|cool|great|nice|hi|hello|hey|yo|bye|goodbye|lol|haha|😀|🙂|😉|👍|👌)\b", re.I)
def is_smalltalk(q: str) -> bool:
    return bool(_SMALLTALK.search(q.strip()))

# 🧠 Prompts

# RAG answer → always short + normal sentence(s)
DOC_PROMPT = PromptTemplate.from_template("""
You are a financial assistant. Use ONLY the provided context.

Answer style:
- Keep it short and normal: 1 sentence for direct facts; up to 2–3 short sentences for open-ended questions.
- If the user asks a specific metric (e.g., Profit Before Tax, Total Assets, Total Liabilities, EPS), include the metric and the value in a natural short sentence (e.g., "Total assets are Rs 2.999T as at 31 Mar 2025.").
- If Total Liabilities aren't given but Total Assets and Total Equity are, you MAY compute Liabilities = Assets − Equity and say it's derived.
- Do NOT confuse commitments/contingencies with liabilities.
- If not available and not derivable, say: "Not available in the provided context."

Context:
{context}

Question: {question}

Answer:
""")

# Rewriter: don't overcook small talk; just pass it through
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Rewrite the follow-up into a standalone question answerable from the documents context.
If the input is simple chit-chat (greetings/thanks/etc.), return it unchanged.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:
""")

# General chat: one short sentence, no RAG
GENERAL_CHAT_PROMPT = PromptTemplate.from_template("""
You are a friendly assistant for SriMethan Holdings (PVT) LTD.
Reply in ONE short sentence, natural and helpful.

User: {user_input}
Assistant:
""")

# 📄 Upload section
st.markdown("<h1 style='text-align:center;'>👑 SRIMETHAN HOLDINGS (PVT) LTD</h1>", unsafe_allow_html=True)
st.markdown("## 📄 Upload Your PDF(s)")
uploaded_files_top = st.file_uploader(
    "Upload here to get started:", type=["pdf"], accept_multiple_files=True
)

# 📄 Sidebar
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

# 📚 Process PDFs
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

    # 🤖 LLM (OpenRouter)
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

    # Build chains
    combine_docs_chain = load_qa_chain(llm, chain_type="stuff", prompt=DOC_PROMPT)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    general_chain = LLMChain(llm=llm, prompt=GENERAL_CHAT_PROMPT)

    st.session_state.qa_chain = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        return_source_documents=False,
    )
    st.session_state.general_chain = general_chain

    st.session_state.vectorstore_ready = True
    st.success("✅ Your files are ready. Start chatting below 👇")

# 💬 Chat UI (user → right, model → left)
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

if st.session_state.vectorstore_ready:
    # History
    for q, a in st.session_state.chat_history:
        render_user(q)
        render_assistant(a)

    # Input
    query = st.chat_input("💬 Type your next question...")
    if query:
        render_user(query)

        # Small talk → answer normally (no RAG), one short sentence
        if is_smalltalk(query):
            resp = st.session_state.general_chain.invoke({"user_input": query})["text"].strip()
            render_assistant(resp)
            st.session_state.chat_history.append((query, resp))
        else:
            response = ""
            placeholder = st.empty()
            placeholder.markdown(
                "<div class='chat-row assistant'><div class='chat-bubble assistant'><b>SriMethan Model :</b> Thinking... 🧠</div></div>",
                unsafe_allow_html=True,
            )

            try:
                for chunk in st.session_state.qa_chain.stream({
                    "question": query,
                    "chat_history": st.session_state.chat_history
                }):
                    token = chunk.get("answer", "")
                    response += token
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

            st.session_state.chat_history.append((query, response.strip()))

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; font-size: 0.9em;'>"
        "Powered by <strong>SriMethan Holdings (PVT) LTD</strong> • © 2025 All rights reserved."
        "</div>",
        unsafe_allow_html=True,
    )
