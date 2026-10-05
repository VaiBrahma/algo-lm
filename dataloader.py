import numpy as np
import torch
import os
from config import get_mac_debug_config

gpt_config, train_config = get_mac_debug_config()

train_path = os.path.join(os.path.dirname(__file__), 'data', 'cpp_corpus', 'train.bin')
val_path = os.path.join(os.path.dirname(__file__), 'data', 'cpp_corpus', 'val.bin')

train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - gpt_config.block_size, (train_config.batch_size, ))
    x = torch.stack([torch.from_numpy(data[i:i+gpt_config.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+gpt_config.block_size+1].astype(np.int64)) for i in ix])
    x, y = x.to(device=train_config.device), y.to(device=train_config.device)
    return x, y

# x, y = get_batch('train')
# print(x[0,], x.shape, y[0, ], y.shape, sep='\n---------------------\n')