"""
Contains separate classes for different embedding models.

Modeled after `train_model.ipynb` tutorial on the Relbench website.
"""
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor

class GloveTextEmbedding:
    """
    Initializes a GloVe embedding model.
    """
    def __init__(self, device: Optional[torch.device
                                       ] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))
