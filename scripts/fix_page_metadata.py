import chromadb

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "langchain"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

batch_size = 500
offset = 0
updated = 0

while True:
    batch = collection.get(
        limit=batch_size,
        offset=offset,
        include=["metadatas"],
    )

    ids = batch["ids"]
    metadatas = batch["metadatas"]

    if not ids:
        break

    for doc_id, metadata in zip(ids, metadatas):
        old_page = metadata.get("page")

        if old_page is None:
            continue

        metadata["pdf_page"] = int(old_page) + 1
        metadata.pop("book_page", None)

        collection.update(
            ids=[doc_id],
            metadatas=[metadata],
        )
        updated += 1

    offset += len(ids)

print(f"Updated metadata for {updated} chunks.")