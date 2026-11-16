from dotenv import load_dotenv
from app.rag.embedding_model import get_embedding_model
from langchain_chroma import Chroma
from functools import lru_cache
load_dotenv()

CHROMA_PATH = "chroma_db"

@lru_cache(maxsize=1)
def get_retriever():

    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    # retriever = vectorstore.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={
    #         "k": 4,
    #         "fetch_k": 10,
    #         "lambda_mult": 0.5
    #     }
    # )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    return retriever


if __name__=="__main__":

    retriever = get_retriever()

    query = input("Question: ")

    docs = retriever.invoke(query)

    for i, doc in enumerate(docs, start=1):

        print(f"\n--- Document {i} ---")
        print(f"Page: {doc.metadata.get('page')}")
        print(f"Source: {doc.metadata.get('source')}")
        print(doc.page_content)