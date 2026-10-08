import torch
from dataclasses import dataclass

@dataclass
class GPTConfig:
    """Hyperparameters that define the neural network architecture."""
    block_size: int = 256
    vocab_size: int = 65 # Default, overwritten by train.py
    n_embd: int = 384
    n_head: int = 6
    n_layers: int = 6
    dropout: float = 0.2

@dataclass
class TrainConfig:
    """Hyperparameters that govern the training loop and hardware."""
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# --- Pre-configured Environments ---

def get_mac_debug_config():
    """Lightweight config to verify the code compiles and runs instantly on MacBook (MPS/CPU)."""
    gpt_cfg = GPTConfig(
        block_size=64,       # Smaller context window saves memory
        n_embd=128,          # Smaller embeddings
        n_head=4,
        n_layers=4,
        dropout=0.0          # No dropout needed for a quick debug run
    )
    
    train_cfg = TrainConfig(
        batch_size=12,       # Small batch size so your Mac doesn't freeze
        max_iters=1500,       # Only run a few iterations to test the pipeline
        eval_interval=20,    # Check loss frequently
        eval_iters=10,
        learning_rate=1e-3,  # Faster learning rate for a tiny model
        device='mps' if torch.backends.mps.is_available() else 'cpu'
    )
    return gpt_cfg, train_cfg


def get_colab_gpu_config():
    """Heavyweight config for actual pre-training on Google Colab."""
    gpt_cfg = GPTConfig(
        block_size=256,      # Full context window
        n_embd=384,          # Deep embeddings (384 / 6 = 64 head size)
        n_head=6,
        n_layers=6,
        dropout=0.2          # Standard dropout to prevent overfitting
    )
    
    train_cfg = TrainConfig(
        batch_size=64,       # Maximize GPU VRAM usage
        max_iters=5000,      # Run a full training cycle
        eval_interval=500,   # Evaluate less often to save time
        eval_iters=200,
        learning_rate=3e-4,  # Standard Transformer learning rate
        device='cuda'        # Force CUDA for the Colab GPUs
    )
    return gpt_cfg, train_cfg