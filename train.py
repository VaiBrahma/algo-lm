import json
from tqdm.auto import tqdm
import torch
from config import GPTConfig, get_mac_debug_config
from model import GPT2 
from dataloader import get_batch

# Setup & Configuration
gpt_config, train_config = get_mac_debug_config()
device = train_config.device

model = GPT2(gpt_config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(train_config.eval_iters)
        for k in range(train_config.eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Training Loop
print('', f"Using device: {device}", "Starting training...", '', sep='\n--------------------\n')

loss_history = {
    "iters": [],
    "train_loss": [],
    "val_loss": []
}
model.train() 

best_val_loss = float('inf')

for iter in tqdm(range(train_config.max_iters), desc="Training algo-lm"):
    
    # Evaluate occasionally
    if iter % train_config.eval_interval == 0 or iter == train_config.max_iters - 1:
        losses = estimate_loss()
        tqdm.write(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss: {losses['val']:.4f}")
        loss_history['iters'].append(iter)
        loss_history['train_loss'].append(losses['train'])
        loss_history['val_loss'].append(losses['val'])
        
        # Save checkpoint if validation loss improves
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': gpt_config,
                'iter': iter,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, 'algo_lm_checkpoint.pth')
            
        # with open('eval/loss_history2.json', 'w') as f:
        #     json.dump(loss_history, f)
            
    # Forward pass
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('', 'Training Complete. Best model saved iteratively.', '', sep='\n-------------------------------\n')