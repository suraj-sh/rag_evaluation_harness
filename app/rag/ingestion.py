from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PDF_PATH = "data/documents/nist_ai_rmf_1.0.pdf"
CHROMA_PATH = "chroma_db"

def create_vector_store():

    # 1. Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    for doc in documents:
        pdf_page = doc.metadata["page"]

        doc.metadata["pdf_page"] = pdf_page
        doc.metadata["book_page"] = pdf_page

    print(f"Loaded {len(documents)} pages")

    # 2. Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # 3. Create embeddings
    embedding_model = HuggingFaceEmbeddings()

    # 4. Store embeddings in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    print("Vector database created")

    return vectorstore


if __name__ == "__main__":
    create_vector_store()