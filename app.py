"""
app.py
Study Agent — AI-powered study assistant using RAG
Upload your lecture PDFs → Ask questions, generate MCQs, summaries, flashcards
"""
 
import os
import streamlit as st
import tempfile
from backend.rag_engine import (
    build_vectorstore,
    load_existing_vectorstore,
    build_qa_chain,
    get_answer,
    generate_mcqs,
    generate_summary,
    generate_flashcards
)
 
# ── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="Study Agent — AI Study Assistant",
    page_icon="📚",
    layout="wide"
)
 
# ── CUSTOM CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0f1117; color: #ffffff; }
    h1 { background: linear-gradient(to right, #a78bfa, #60a5fa);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 800; }
    .answer-box { background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.3);
                  border-left: 4px solid #a78bfa; border-radius: 10px;
                  padding: 1.2rem; margin: 0.8rem 0; }
    .source-box { background: rgba(96,165,250,0.06); border: 1px solid rgba(96,165,250,0.2);
                  border-radius: 8px; padding: 0.8rem; margin: 0.4rem 0;
                  font-size: 0.85rem; color: #94a3b8; }
    .tag { background: rgba(167,139,250,0.2); color: #c4b5fd;
           padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; }
    .stButton > button { background: linear-gradient(135deg, #7c3aed, #4f46e5);
                         color: white; border: none; border-radius: 8px;
                         font-weight: 600; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-1px);
                                box-shadow: 0 6px 20px rgba(124,58,237,0.4); }
</style>
""", unsafe_allow_html=True)
 
 
# ── SESSION STATE INIT ──────────────────────────────────
if "qa_chain"       not in st.session_state: st.session_state.qa_chain       = None
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "docs_loaded"    not in st.session_state: st.session_state.docs_loaded    = False
if "uploaded_names" not in st.session_state: st.session_state.uploaded_names = []
 
 
# ── HEADER ──────────────────────────────────────────────
st.title("📚 Study Agent")
st.markdown("**Upload your lecture PDFs → Ask anything → Generate MCQs, Summaries & Flashcards**")
st.divider()
 
 
# ── SIDEBAR: PDF UPLOAD ─────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Study Material")
    st.caption("Supported: PDF files (lecture notes, textbooks, slides)")
 
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once"
    )
 
    if uploaded_files:
        if st.button("🔄 Process PDFs", use_container_width=True):
            with st.spinner("Reading PDFs and building knowledge base... (first time takes ~30 sec)"):
                try:
                    # Save uploaded files to temp directory
                    temp_paths = []
                    with tempfile.TemporaryDirectory() as tmpdir:
                        for f in uploaded_files:
                            path = os.path.join(tmpdir, f.name)
                            with open(path, "wb") as out:
                                out.write(f.read())
                            temp_paths.append(path)
 
                        # Build vector store
                        vectorstore = build_vectorstore(temp_paths)
 
                    # Build QA chain
                    st.session_state.qa_chain       = build_qa_chain(vectorstore)
                    st.session_state.docs_loaded    = True
                    st.session_state.chat_history   = []
                    st.session_state.uploaded_names = [f.name for f in uploaded_files]
 
                    st.success(f"✅ {len(uploaded_files)} PDF(s) processed!")
 
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")
 
    # Show loaded docs
    if st.session_state.docs_loaded:
        st.divider()
        st.markdown("**📄 Loaded Documents:**")
        for name in st.session_state.uploaded_names:
            st.markdown(f'<span class="tag">📄 {name}</span>', unsafe_allow_html=True)
 
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            st.session_state.qa_chain       = None
            st.session_state.docs_loaded    = False
            st.session_state.chat_history   = []
            st.session_state.uploaded_names = []
            # Clear ChromaDB
            import shutil
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")
            st.rerun()
 
    st.divider()
    st.markdown("**🔑 Powered by:**")
    st.caption("🧠 Groq Llama3-70B (free)")
    st.caption("🗃️ ChromaDB (local)")
    st.caption("🔍 all-MiniLM-L6-v2 embeddings (free)")
    st.caption("📄 PDFPlumber")
 
 
# ── MAIN AREA ───────────────────────────────────────────
if not st.session_state.docs_loaded:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1** 📂\nUpload your lecture PDFs in the sidebar")
    with col2:
        st.info("**Step 2** ⚡\nClick 'Process PDFs' to build the knowledge base")
    with col3:
        st.info("**Step 3** 🤖\nAsk questions, generate MCQs, summaries & flashcards")
 
    st.markdown("---")
    st.markdown("### 💡 What can this agent do?")
    examples = [
        ("💬 Answer Questions", "Ask anything from your PDFs — it finds the exact answer with source references"),
        ("📝 Generate MCQs",    "Auto-generate multiple choice questions for exam prep on any topic"),
        ("📋 Summarize Topics", "Get structured summaries with key concepts and exam points"),
        ("🃏 Create Flashcards","Generate flashcard-style Q&A pairs for quick revision"),
    ]
    cols = st.columns(4)
    for col, (title, desc) in zip(cols, examples):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)
 
else:
    # ── TABS ──────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📝 MCQ Generator", "📋 Summary", "🃏 Flashcards"])
 
    # ── TAB 1: CHAT ────────────────────────────────────
    with tab1:
        st.subheader("Ask anything from your study material")
 
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**🧑 You:** {msg['content']}")
            else:
                st.markdown(f'<div class="answer-box">🤖 <b>Agent:</b><br>{msg["content"]}</div>',
                            unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander("📎 Source chunks used"):
                        for s in msg["sources"]:
                            st.markdown(
                                f'<div class="source-box"><b>{s["file"]}</b><br>{s["preview"]}</div>',
                                unsafe_allow_html=True
                            )
 
        # Question input
        st.divider()
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_question = st.text_input(
                "Your question",
                placeholder="e.g. Explain backpropagation. What is gradient descent? Summarize chapter 3.",
                label_visibility="collapsed"
            )
        with col_btn:
            ask_clicked = st.button("Ask →", use_container_width=True)
 
        # Quick example questions
        st.caption("Try: &nbsp;"
                   "**What are the main topics covered?** &nbsp;|&nbsp; "
                   "**Explain [concept] in simple terms** &nbsp;|&nbsp; "
                   "**What are the key formulas?**")
 
        if ask_clicked and user_question.strip():
            with st.spinner("Searching your notes and generating answer..."):
                result = get_answer(st.session_state.qa_chain, user_question)
 
            # Save to history
            st.session_state.chat_history.append({"role": "user",    "content": user_question})
            st.session_state.chat_history.append({"role": "assistant","content": result["answer"],
                                                   "sources": result["sources"]})
            st.rerun()
 
        # Clear chat
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
 
    # ── TAB 2: MCQ GENERATOR ───────────────────────────
    with tab2:
        st.subheader("📝 Auto-Generate MCQs for Exam Prep")
        col1, col2 = st.columns([3, 1])
        with col1:
            mcq_topic = st.text_input("Topic for MCQs",
                                       placeholder="e.g. Neural Networks, Gradient Descent, Transformers")
        with col2:
            num_mcqs = st.selectbox("Number of MCQs", [3, 5, 8, 10], index=1)
 
        if st.button("⚡ Generate MCQs", use_container_width=False):
            if mcq_topic.strip():
                with st.spinner(f"Generating {num_mcqs} MCQs on '{mcq_topic}'..."):
                    mcqs = generate_mcqs(st.session_state.qa_chain, mcq_topic, num_mcqs)
                st.markdown(f'<div class="answer-box">{mcqs.replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True)
                st.download_button("💾 Download MCQs", mcqs,
                                   file_name=f"mcqs_{mcq_topic.replace(' ','_')}.txt")
            else:
                st.warning("Please enter a topic first.")
 
    # ── TAB 3: SUMMARY ─────────────────────────────────
    with tab3:
        st.subheader("📋 Get Structured Topic Summaries")
        summary_topic = st.text_input("Topic to Summarize",
                                       placeholder="e.g. Convolutional Neural Networks, Attention Mechanism")
 
        if st.button("📋 Generate Summary", use_container_width=False):
            if summary_topic.strip():
                with st.spinner(f"Summarizing '{summary_topic}' from your notes..."):
                    summary = generate_summary(st.session_state.qa_chain, summary_topic)
                st.markdown(f'<div class="answer-box">{summary.replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True)
                st.download_button("💾 Download Summary", summary,
                                   file_name=f"summary_{summary_topic.replace(' ','_')}.txt")
            else:
                st.warning("Please enter a topic first.")
 
    # ── TAB 4: FLASHCARDS ──────────────────────────────
    with tab4:
        st.subheader("🃏 Generate Flashcards for Quick Revision")
        col1, col2 = st.columns([3, 1])
        with col1:
            flash_topic = st.text_input("Topic for Flashcards",
                                         placeholder="e.g. Deep Learning Basics, NLP Terms, ML Algorithms")
        with col2:
            num_cards = st.selectbox("Number of Cards", [5, 8, 10, 15], index=1)
 
        if st.button("🃏 Generate Flashcards", use_container_width=False):
            if flash_topic.strip():
                with st.spinner(f"Creating {num_cards} flashcards on '{flash_topic}'..."):
                    cards = generate_flashcards(st.session_state.qa_chain, flash_topic, num_cards)
                st.markdown(f'<div class="answer-box">{cards.replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True)
                st.download_button("💾 Download Flashcards", cards,
                                   file_name=f"flashcards_{flash_topic.replace(' ','_')}.txt")
            else:
                st.warning("Please enter a topic first.")
 









