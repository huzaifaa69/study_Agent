"""
backend/rag_engine.py
Core RAG logic - rewritten for LangChain 0.2+ (no deprecated chains)
"""

import os
import pdfplumber
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
EMBED_MODEL    = "all-MiniLM-L6-v2"
LLM_MODEL      = "llama-3.3-70b-versatile"
CHROMA_PATH    = "chroma_db"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
TOP_K_RESULTS  = 4


def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n\n--- Page {page_num + 1} ---\n{text}"
    return full_text


def build_vectorstore(pdf_paths: list, collection_name: str = "study_docs") -> Chroma:
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    for pdf_path in pdf_paths:
        raw_text = extract_text_from_pdf(pdf_path)
        filename  = os.path.basename(pdf_path)
        chunks = splitter.create_documents(
            texts=[raw_text],
            metadatas=[{"source": filename}]
        )
        all_docs.extend(chunks)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=collection_name
    )
    vectorstore.persist()
    return vectorstore


def load_existing_vectorstore(collection_name: str = "study_docs") -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=collection_name
    )


def build_qa_chain(vectorstore: Chroma) -> dict:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.3,
        max_tokens=1024
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )
    return {"llm": llm, "retriever": retriever}


def get_answer(chain: dict, question: str, chat_history: list = []) -> dict:
    llm       = chain["llm"]
    retriever = chain["retriever"]
    docs      = retriever.get_relevant_documents(question)
    context   = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        SystemMessage(content=f"""You are a helpful study assistant.
Answer questions based ONLY on the following study material context.
If the answer is not in the context, say "I couldn't find this in the uploaded documents."
Always be clear, concise and helpful for exam preparation.

CONTEXT FROM STUDY MATERIAL:
{context}
""")
    ]
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    sources  = []
    for doc in docs:
        info = {"file": doc.metadata.get("source", "Unknown"), "preview": doc.page_content[:200] + "..."}
        if info not in sources:
            sources.append(info)
    return {"answer": response.content, "sources": sources}


def generate_mcqs(chain: dict, topic: str, num: int = 5) -> str:
    llm       = chain["llm"]
    retriever = chain["retriever"]
    docs      = retriever.get_relevant_documents(topic)
    context   = "\n\n".join([doc.page_content for doc in docs])
    prompt    = f"""Based on this study material, generate {num} multiple choice questions about '{topic}'.

STUDY MATERIAL:
{context}

Format each question exactly like this:
Q1. [Question text]
A) [Option]  B) [Option]  C) [Option]  D) [Option]
Answer: [Correct letter]
Explanation: [Brief explanation]"""
    return llm.invoke([HumanMessage(content=prompt)]).content


def generate_summary(chain: dict, topic: str) -> str:
    llm       = chain["llm"]
    retriever = chain["retriever"]
    docs      = retriever.get_relevant_documents(topic)
    context   = "\n\n".join([doc.page_content for doc in docs])
    prompt    = f"""From this study material, provide a clear summary of '{topic}'.

STUDY MATERIAL:
{context}

Structure as:
1. Key Definition
2. Core Concepts (3-5 bullet points)
3. Important Formulas or Rules (if any)
4. Common Exam Points to Remember"""
    return llm.invoke([HumanMessage(content=prompt)]).content


def generate_flashcards(chain: dict, topic: str, num: int = 8) -> str:
    llm       = chain["llm"]
    retriever = chain["retriever"]
    docs      = retriever.get_relevant_documents(topic)
    context   = "\n\n".join([doc.page_content for doc in docs])
    prompt    = f"""Create {num} flashcard Q&A pairs about '{topic}'.

STUDY MATERIAL:
{context}

Format each as:
FRONT: [Short question or term]
BACK: [Concise answer or definition]
---"""
    return llm.invoke([HumanMessage(content=prompt)]).content
