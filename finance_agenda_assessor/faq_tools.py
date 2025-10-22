import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PDF_PATH = "../FAQ_assesspr_v1.1.pdf"

DATABASE_URL = os.getenv("DATABASE_URL")  


def get_faq_context(question: str) -> str:
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunck_size=700, chunck_overlap=150)
    chunks = splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004", google_api_key = DATABASE_URL, transport="rest")

    db = FAISS.from_documents(chunks, embeddings)
    results = db.similarity_search(question, k=6)

    context_text = "\n\n".join([r.page_content for r in results])
    return context_text