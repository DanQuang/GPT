import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union
from dataloader.tokenizer import SimpleTokenizer


class GPTDataset(Dataset):
    def __init__(self,
                 txt: str,
                 tokenizer,
                 max_length,
                 stride: int = 1):
        # Initialize the parent Dataset class
        super(GPTDataset, self).__init__()

        # Using tiktoken BPE for default
        self.token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Lists to hold input and target IDs
        self.input_ids = []
        self.target_ids = []

        # Use a sliding window to create overlapping sequences of max_length
        for i in range(0, len(self.token_ids) - max_length, stride):
            # Extract a chunk of token IDs for the input
            input_chunk = self.token_ids[i:i + max_length]
            # Extract the corresponding target chunk (next tokens)
            target_chunk = self.token_ids[i + 1: i + max_length + 1]
            # Append the input and target chunks as tensors
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))  

    def __len__(self):
        # Return the total number of input-output pairs in the dataset
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return the input and target tensors at the specified index
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt= txt,
                         tokenizer= tokenizer,
                         max_length= max_length,
                         stride= stride)
    dataloader = DataLoader(dataset= dataset,
                            batch_size= batch_size,
                            shuffle= shuffle)
    
    return dataloader