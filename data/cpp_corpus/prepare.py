import os
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

input_file_path = os.path.join(os.path.dirname(__file__), 'raw_code.txt')
with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()
    
data = enc.encode(text, allowed_special={'<|endoftext|>'})

n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

# Convert to 16-bit integers (since vocab size 50,257 < 65,535)
train_ids = np.array(train_data, dtype=np.uint16)
val_ids = np.array(val_data, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))