# algo-lm
An educational, bottom-up implementation of a decoder-only Generative Pre-trained Transformer (GPT) built from scratch in PyTorch. 

### Highlights
- **From Scratch:** Implements token embeddings, causal multi-head self-attention, feed-forward networks, and residual connections without relying on high-level transformer abstractions.
- **Hardware Agnostic:** Configured to run natively on Apple Silicon (Metal Performance Shaders / MPS) and standard CUDA/CPU backends.
- **Phased Evolution:** 
  - `v0.1`: Bigram language model baseline
  - `v1.0`: Decoder-only Transformer trained on Tiny Shakespeare
  - `v2.0`: Instruction-tuned model for C++ competitive programming templates
