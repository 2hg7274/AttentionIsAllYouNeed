import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_path = "davidkim205/komt-mistral-7b-v1"
csv_path = "/home/ubuntu/korean-parallel-corpora/train.csv"

batch_size = 16
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

init_lr = 1e-5
weight_decay = 5e-4
adam_eps = 5e-9
factor = 0.9
patience = 10
epoch = 1000
clip = 1.0
warmup = 100
inf = float('inf')