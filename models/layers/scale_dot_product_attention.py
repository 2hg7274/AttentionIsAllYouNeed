import math
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query: given sentence that we focused on (decoder)
    Key: every sentence to check relationship with Query (encoder)
    Value: every sentence same with Key (encoder)
    """

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()


        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3) 
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 2. apply masking 
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0,1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score



