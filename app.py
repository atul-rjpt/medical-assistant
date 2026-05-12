import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Agentic AI Medical Assistant", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .main-header { font-size:2.4rem; font-weight:700; background:linear-gradient(135deg,#1565C0,#0288D1);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-align:center; }
    .sub-header { font-size:0.95rem; color:#607D8B; text-align:center; margin-bottom:1.5rem; }
    .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; margin:2px; }
    .badge-blue  { background:#E3F2FD; color:#1565C0; }
    .badge-green { background:#E8F5E9; color:#2E7D32; }
    .badge-red   { background:#FFEBEE; color:#C62828; }
    .stat-box { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:1rem; text-align:center; }
    .stat-num  { font-size:1.8rem; font-weight:700; color:#1565C0; }
    .stat-label{ font-size:0.8rem; color:#607D8B; }
    .warning-box { background:#FFF8E1; border-left:4px solid #F9A825; border-radius:6px; padding:0.75rem 1rem; font-size:0.85rem; color:#5D4037; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏥 Agentic AI Medical Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><span class="badge badge-blue">RAG Pipeline</span><span class="badge badge-blue">FAISS Retrieval</span><span class="badge badge-blue">DeepSeek R1</span><span class="badge badge-green">Context-Grounded</span><span class="badge badge-red">Hallucination-Reduced</span></div>', unsafe_allow_html=True)

MEDICAL_PROMPT = PromptTemplate(
    template="""You are an Agentic AI Medical Assistant powered by DeepSeek R1.
Answer questions strictly based on the uploaded medical documents.

STRICT RULES:
1. ONLY use information found in the provided context. Never hallucinate.
2. If the answer is not in the context, say: "This information is not found in the uploaded medical documents."
3. Cite the document/section when possible.
4. For critical medical decisions always add: "⚠️ Please consult a licensed physician."

Context from Medical Documents:
{context}

Question: {input}

Answer (strictly based on uploaded documents):""",
    input_variables=["context", "input"]
)

for key, default in {"vectorstore":None,"qa_chain":None,"chat_history":[],"doc_stats":{},"processed":False}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    try:
        default_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except Exception:
        default_key = os.getenv("GROQ_API_KEY", "")

    groq_api_key = st.text_input("🔑 Groq API Key", type="password", value=default_key, help="Free at console.groq.com")
    st.markdown("---")
    st.markdown("## 🤖 Model Settings")
    model_choice = st.selectbox("LLM Model", ["llama-3.3-70b-versatile","llama-3.1-8b-instant","gemma2-9b-it","compund-beta"])
    temperature  = st.slider("Temperature", 0.0, 1.0, 0.05, 0.05)
    top_k_docs   = st.slider("Retrieved Chunks (k)", 2, 8, 4)
    st.markdown("---")
    st.markdown("## 📄 Upload Medical Documents")
    uploaded_files = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)
    process_btn    = st.button("🔄 Process Documents", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown('<div class="warning-box">⚠️ For informational use only. Always consult a licensed physician.</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def process_pdfs(files):
    all_docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = f.name
        all_docs.extend(docs)
        os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(all_docs), len(all_docs)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(llm, retriever):
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | MEDICAL_PROMPT
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

if process_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    elif not groq_api_key:
        st.sidebar.error("Please enter your Groq API Key.")
    else:
        with st.spinner("📊 Processing medical documents..."):
            try:
                progress = st.progress(0, text="Loading PDFs...")
                chunks, total_pages = process_pdfs(uploaded_files)
                progress.progress(33, text="Building FAISS index...")
                embeddings  = load_embeddings()
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.vectorstore = vectorstore
                progress.progress(66, text="Initialising LLM chain...")
                llm = ChatGroq(api_key=groq_api_key, model_name=model_choice, temperature=temperature, max_tokens=1024)
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k_docs})
                st.session_state.qa_chain  = build_chain(llm, retriever)
                st.session_state.processed = True
                st.session_state.doc_stats = {"files":len(uploaded_files),"pages":total_pages,"chunks":len(chunks),"model":model_choice.split("-")[0].upper()}
                progress.progress(100, text="Done!")
                time.sleep(0.4)
                progress.empty()
                st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} document(s).")
            except Exception as e:
                st.error(f"Processing failed: {e}")

if st.session_state.processed:
    s = st.session_state.doc_stats
    for col, num, label in zip(st.columns(4), [s["files"],s["pages"],s["chunks"],s["model"]], ["Documents","Pages","Chunks","Model"]):
        col.markdown(f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("")

col_title, col_clear = st.columns([5,1])
with col_title:
    st.markdown("### 💬 Chat with Your Medical Records")
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"]=="user" else "🏥"):
        st.markdown(msg["content"])
        if msg["role"]=="assistant" and msg.get("sources"):
            with st.expander("📚 Retrieved Sources"):
                for src in msg["sources"]:
                    st.caption(f"📄 **{src['file']}** — Page {src['page']}")

placeholder = "Ask about diagnoses, lab results, medications..." if st.session_state.processed else "Upload and process documents first..."

if user_input := st.chat_input(placeholder, disabled=not st.session_state.processed):
    st.session_state.chat_history.append({"role":"user","content":user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)
    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("🔍 Searching medical records..."):
            try:
                result      = st.session_state.qa_chain.invoke(user_input)
                answer      = result["answer"]
                source_docs = result.get("context", [])
                st.markdown(answer)
                sources, seen = [], set()
                for doc in source_docs:
                    key = (doc.metadata.get("source_file","Unknown"), doc.metadata.get("page",0))
                    if key not in seen:
                        seen.add(key)
                        sources.append({"file":doc.metadata.get("source_file","Unknown"),"page":doc.metadata.get("page",0)+1})
                if sources:
                    with st.expander("📚 Retrieved Sources"):
                        for src in sources:
                            st.caption(f"📄 **{src['file']}** — Page {src['page']}")
                st.session_state.chat_history.append({"role":"assistant","content":answer,"sources":sources})
            except Exception as e:
                err = f"❌ Error: {str(e)}"
                st.error(err)
                st.session_state.chat_history.append({"role":"assistant","content":err,"sources":[]})

if not st.session_state.processed:
    st.markdown("---")
    with st.expander("📖 How to Use", expanded=True):
        st.markdown("""
| Step | Action |
|------|--------|
| 1️⃣  | Enter your **Groq API Key** in the sidebar |
| 2️⃣  | Upload **medical PDF(s)** |
| 3️⃣  | Click **Process Documents** |
| 4️⃣  | Ask questions in the chat |

**Example Questions:**
- *"What medications is the patient currently on?"*
- *"What were the CBC blood test results?"*
- *"Are there any drug allergies mentioned?"*

> ⚠️ For informational purposes only. Always consult a licensed physician.
        """)