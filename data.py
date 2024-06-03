from conf import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.model_max_length = max_len
tokenizer.pad_token = tokenizer.eos_token

class CustomDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, encoding='utf-8')

        self.src = self.df['en']
        self.trg = self.df['ko']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        src = tokenizer(self.src.iloc[idx], padding='max_length', truncation=True, return_tensors='pt')
        trg = tokenizer(self.trg.iloc[idx], padding='max_length', truncation=True, return_tensors='pt')
        
        src_ids = src['input_ids'].squeeze(0)
        trg_ids = trg['input_ids'].squeeze(0)
        
        return src_ids, trg_ids


dataset = CustomDataset(csv_path)
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
val_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - val_size


train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(valid_dataset)}")
print(f"Testing Data Size : {len(test_dataset)}")

train_iter = DataLoader(train_dataset, batch_size=batch_size)
valid_iter = DataLoader(valid_dataset, batch_size=batch_size)
test_iter = DataLoader(test_dataset, batch_size=batch_size)
