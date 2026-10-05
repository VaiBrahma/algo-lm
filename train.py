import torch
import tiktoken
from config import GPTConfig, get_mac_debug_config
from model import GPT2 
from dataloader import get_batch

gpt_config, train_config = get_mac_debug_config()
device = train_config.device

enc = tiktoken.get_encoding('gpt2')

model = GPT2(gpt_config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

@torch.no_grad() # Tells PyTorch not to store gradients (saves memory)
def estimate_loss():
    out = {}
    model.eval() # Turn OFF dropout for evaluation

    for split in ['train', 'val']:
        # We take average loss over eval_iters (e.g., 200 batches) to reduce noise
        losses = torch.zeros(train_config.eval_iters)
        
        for k in range(train_config.eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
            
        out[split] = losses.mean().item()
        
    model.train() # Turn dropout back ON for training
    return out

# 2. Training Loop
# CRITICAL: Turn on Dropout and LayerNorm training behavior
model.train() 

for iter in range(train_config.max_iters):
    
    if iter % train_config.eval_interval == 0 or iter == train_config.max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss: {losses['val']:.4f}")
        
    # Forward pass
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    # Backpropagation
    optimizer.zero_grad(set_to_none=True) # set_to_none=True is faster than default
    loss.backward()
    optimizer.step()
    
print('\nTraining Complete.')

# 3. Generation Test (algo-lm prompt)
model.eval() # Turn OFF Dropout for generation

context_text = 'int main() {'
context_tokens = enc.encode(context_text) # tiktoken returns a list of ints
idx = torch.tensor([context_tokens], dtype=torch.long, device=device) # Shape: (1, T)

with torch.no_grad():
    generated_indices = model.generate(idx, max_new_tokens=250)[0].tolist()
    
print(enc.decode(generated_indices))

# 4. Save Model Checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': gpt_config, # Save the config so you know the exact shape later
}
torch.save(checkpoint, 'algo_lm_checkpoint.pth')
print("\nModel checkpoint saved successfully!")