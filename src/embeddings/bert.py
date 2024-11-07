"""
Contains separate classes for different embedding models.

Modeled after `train_model.ipynb` tutorial on the Relbench website.
"""
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor

class BertTextEmbedding:
    """
    Initializes a BERT embedding model.
    """
    def __init__(self, device: Optional[torch.device
                                       ] = None):
        self.model = SentenceTransformer(
            "google-bert/bert-base-uncased",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))