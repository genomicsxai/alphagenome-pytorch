import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from alphagenome_pytorch.model import AlphaGenome

def main():
    # Create model
    print("Initializing model...")
    # NOTE: AlphaGenome now has standard heads configured internally
    model = AlphaGenome(num_organisms=2) 
    
    # Load weights
    weights_path = "model.pth"
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(missing) > 0:
            print("Missing keys:", missing[:5], "..." if len(missing)>5 else "")
    else:
        print(f"Warning: {weights_path} not found. Running with random weights.")

    model.eval()

    # Create dummy input
    # (Batch, Sequence, Channels)
    # 131072 is standard window
    inputs = torch.randn(1, 131072, 4)
    organism_index = torch.tensor([0], dtype=torch.long)
    
    print(f"Input shape: {inputs.shape}")

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(inputs, organism_index)

    print("Output keys:", outputs.keys())
    for k, v in outputs.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  Resolution {sub_k}: {sub_v.shape}")
        else:
            print(f"{k}: {v.shape}")

if __name__ == "__main__":
    main()
