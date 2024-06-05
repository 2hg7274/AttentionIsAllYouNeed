from torch import nn
from data import *

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        """
        :param vocab_size: size of the dictionary of embeddings
        :param d_model: the size of each embedding vector
        """
        super().__init__(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=tokenizer.pad_token_id)