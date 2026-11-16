from functools import lru_cache
import torch
from langchain_huggingface import HuggingFaceEmbeddings

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and reuse one embedding model per Python process."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )