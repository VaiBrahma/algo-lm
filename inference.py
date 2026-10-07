import torch
import tiktoken
from model import GPT2
from config import GPTConfig # Required so torch.load can unpickle the config object

def fprint(msg):
    separater = '\n' + '-' * len(msg) + '\n'
    print('', msg, '', sep=separater)

def load_model(checkpoint_path, device='cpu'):
    fprint(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = GPT2(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() 
    return model

def generate_cpp(model, prompt, max_new_tokens=250, device='cpu'):
    enc = tiktoken.get_encoding('gpt2')
    context_tokens = enc.encode(prompt)
    idx = torch.tensor([context_tokens], dtype=torch.long, device=device) # (1, T)
    
    with torch.no_grad():
        generated_indices = model.generate(idx, max_new_tokens)[0].tolist()
        
    return enc.decode(generated_indices)

if __name__ == '__main__':
    device = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    fprint(f"Using device: {device}")
    try:
        model = load_model('algo_lm_colab.pth', device)
        fprint("Model loaded successfully. Ready for generation.")
        fprint(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except FileNotFoundError:
        fprint("Error: 'algo_lm_colab.pth' not found. Run train.py first.")
        exit(1)

    # Interactive testing loop
    fprint("Type a C++ prompt (or type 'quit' to exit):")
    prompt = ""
    while len(prompt) == 0:
        prompt = input("\n>>> ")
        if prompt.lower() in ['quit', 'exit']:
            break
            
        fprint("--- Generating ---")
        output = generate_cpp(model, prompt, max_new_tokens=300, device=device)
        print(output)
        fprint("--------------------------------")