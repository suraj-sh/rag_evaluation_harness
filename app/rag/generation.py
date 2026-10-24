import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"


def generate_answer(question: str, documents: list) -> str:
    """Generate an answer grounded in retrieved document chunks."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing from the environment.")

    context_parts = []

    for doc in documents:
        page = doc.metadata.get("pdf_page")

        if page is None:
            raw_page = doc.metadata.get("page")
            page = raw_page + 1 if isinstance(raw_page, int) else "unknown"
        source = doc.metadata.get("source", "unknown")

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are an assistant answering questions about the NIST AI Risk
Management Framework.

Answer the question using only the retrieved context below.

Rules:
- Do not invent facts that are not supported by the context.
- If the context does not contain enough information, say so clearly.
- Cite relevant pages using the format [Page N], using the page
  numbers provided in the context.
- Give a direct, useful answer.

Retrieved context:
{context}

Question:
{question}

Answer:
"""

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=api_key,
        timeout=60,
        max_retries=2
    )

    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        print(f"Gemini generation failed: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "The AI service is temporarily unavailable. "
            "Please wait a moment and try again."
        ) from exc

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(text_parts)

    return str(content)