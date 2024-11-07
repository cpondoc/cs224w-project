"""
Contains separate classes for different embedding models.

Modeled after `train_model.ipynb` tutorial on the Relbench website.
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor


class CustomTextEmbedding:
    """
    Initializes a Custom embedding model, using HuggingFace models.
    """

    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))
