from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
load_dotenv()

CHROMA_PATH = "chroma_db"

def get_retriever():

    embedding_model = HuggingFaceEmbeddings()

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