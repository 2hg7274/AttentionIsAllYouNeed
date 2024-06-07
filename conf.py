import torch

# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path
tokenizer_path = "davidkim205/komt-mistral-7b-v1"
csv_path = "/home/ubuntu/korean-parallel-corpora/train.csv"

# model configurations
batch_size = 16
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.2

# Trainig configurations
epoch = 100
gradient_accumulation_steps = 4
init_lr = 2e-4 
weight_decay = 5e-4
adam_eps = 5e-9
factor = 0.9
patience = 2
clip = 1.0
warmup = 5
T_mult = 2
eta_min = 0
inf = float('inf')