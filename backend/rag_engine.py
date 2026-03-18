import os
import pdfplumber
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
EMBED_MODEL    = "all-MiniLM-L6-v2"
LLM_MODEL      = "llama-3.3-70b-versatile"
CHROMA_PATH    = "chroma_db"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
TOP_K_RESULTS  = 4

def extract_text_from_pdf(pdf_path):
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    full_text += f"\n\n--- Page {page_num + 1} ---\n{text}"
    except Exception:
        pass
    if not full_text.strip():
        full_text = "This PDF has no extractable text. Please use a text-based PDF."
    return full_text

def build_vectorstore(pdf_paths, collection_name="study_docs"):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for pdf_path in pdf_paths:
        raw_text = extract_text_from_pdf(pdf_path)
        filename = os.path.basename(pdf_path)
        chunks = splitter.create_documents(texts=[raw_text], metadatas=[{"source": filename}])
        all_docs.extend(chunks)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    vectorstore = Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory=CHROMA_PATH, collection_name=collection_name)
    vectorstore.persist()
    return vectorstore

def load_existing_vectorstore(collection_name="study_docs"):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings, collection_name=collection_name)

def build_qa_chain(vectorstore):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.3, max_tokens=1024)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K_RESULTS})
    return {"llm": llm, "retriever": retriever}

def get_answer(chain, question, chat_history=[]):
    llm = chain["llm"]
    retriever = chain["retriever"]
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    messages = [SystemMessage(content=f"You are a helpful study assistant. Answer based ONLY on this study material. If the answer is not in the context, say so.\n\nCONTEXT:\n{context}")]
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    sources = []
    for doc in docs:
        info = {"file": doc.metadata.get("source", "Unknown"), "preview": doc.page_content[:200] + "..."}
        if info not in sources:
            sources.append(info)
    return {"answer": response.content, "sources": sources}

def generate_mcqs(chain, topic, num=5):
    llm = chain["llm"]
    retriever = chain["retriever"]
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Based on this study material, generate {num} MCQs about '{topic}'.\n\nSTUDY MATERIAL:\n{context}\n\nFormat:\nQ1. [Question]\nA) B) C) D)\nAnswer: [Letter]\nExplanation: [Brief]"
    return llm.invoke([HumanMessage(content=prompt)]).content

def generate_summary(chain, topic):
    llm = chain["llm"]
    retriever = chain["retriever"]
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Summarize '{topic}' from this material:\n\n{context}\n\nStructure as:\n1. Key Definition\n2. Core Concepts\n3. Formulas\n4. Exam Points"
    return llm.invoke([HumanMessage(content=prompt)]).content

def generate_flashcards(chain, topic, num=8):
    llm = chain["llm"]
    retriever = chain["retriever"]
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Create {num} flashcards about '{topic}'.\n\nMATERIAL:\n{context}\n\nFormat:\nFRONT: [question]\nBACK: [answer]\n---"
    return llm.invoke([HumanMessage(content=prompt)]).content
