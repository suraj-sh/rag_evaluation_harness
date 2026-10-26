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
- Answer using only the retrieved context. Do not invent unsupported facts.
- Address every part of the question and include all relevant facts,
  categories, qualifications, exceptions, and limitations stated in the context.
- For questions asking what information should be considered, identify each
  distinct information category explicitly rather than grouping categories
  into vague or general statements.
- For questions asking for a list or multiple categories, check the retrieved
  context for every relevant item before finalizing the answer.
- Explicitly distinguish intended purposes and beneficial uses from potential
  harmful or unintended uses, where the context discusses them.
- When discussing impacts, state the types of impacts and who or what may
  be affected, as described in the context.
- Preserve important qualifications. For example, if the context states that
  something can occur without a particular intention, do not omit that
  qualification when it answers the question.
- If the retrieved context does not contain enough information to answer a part of the question,
  state that limitation clearly, but still answer every other part that is supported by the context.
- When a question contains an unsupported assumption or asks for a guarantee, percentage,
  or exact figure that the context does not provide, explicitly state that the context does not establish that claim.
  Do not stop there if the context contains relevant related information that helps answer the question.
- Distinguish between what the source explicitly states, what it does not establish,
  and what it recommends or identifies as future work.
  Cite the relevant pages for each supported statement.
- When the context explicitly recommends that organizations periodically
  evaluate a framework's effectiveness, include that recommendation when
  answering questions about the framework's effectiveness or expected outcomes.
- When the question or expected information concerns documentation, explicitly
  identify what information should be documented, if the context supports it.
- Cite relevant pages using the format [Page N], using the page numbers
  provided in the context.
- Give a direct, well-organized answer without unnecessary repetition.
- When answering questions about MEASURE, explicitly explain that measurement
  methods and metrics should be appropriate to the context of use and the
  specific risk or trustworthiness characteristic being assessed. Include
  quantitative, qualitative, and mixed methods where supported by the context.
  Connect measurement to risks identified during MAP, prioritize significant
  risks, and document risks or trustworthiness characteristics that will not
  or cannot be measured. Do not imply every risk must be quantified.
- When answering questions about MANAGE, explicitly explain how risks are
  prioritized and how responses are carried out. Distinguish prioritizing
  documented risks based on impact, likelihood, and available resources from
  developing, planning, documenting, and implementing responses to high-priority
  risks. Include supported response options such as mitigating, transferring,
  avoiding, or accepting risks. Explain that organizations consider resources
  needed to manage risks and viable non-AI alternatives. Do not invent a fixed
  prioritization formula, mandatory allocation procedure, or unsupported
  implementation requirement.

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