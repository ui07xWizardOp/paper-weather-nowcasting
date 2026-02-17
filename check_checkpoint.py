import torch
import os
import sys

def check_weights(file_path):
    print(f"🔍 Inspecting checkpoint: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    try:
        # Load to CPU to be safe
        checkpoint = torch.load(file_path, map_location='cpu')
        
        # dynamic handling if it's a full checkpoint dict or just state_dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        else:
            state_dict = checkpoint
            print("   (Format: direct state_dict)")

        n_params = 0
        n_corrupt = 0
        
        print("   Scanning parameters...", end="", flush=True)
        
        for name, param in state_dict.items():
            if not torch.is_tensor(param):
                continue
                
            n_params += 1
            if torch.isnan(param).any():
                print(f"\n   ❌ NaN detected in layer: {name}")
                n_corrupt += 1
            elif torch.isinf(param).any():
                print(f"\n   ❌ Inf detected in layer: {name}")
                n_corrupt += 1
        
        print(" Done.")

        if n_corrupt > 0:
            print(f"\n⚠️ Verdict: CORRUPTED ({n_corrupt} layers affected)")
            print("Action: Delete this file and resume from 'best_model.pth' or previous backup.")
            return False
        else:
            print("\n✅ Verdict: HEALTHY. Weights are valid numbers.")
            print("Note: If loss was -inf, the optimizer/scaler likely skipped the update, preserving weights.")
            return True
            
    except Exception as e:
        print(f"\n❌ Error loading file: {e}")
        return False

if __name__ == "__main__":
    # Default paths to check
    default_path = '/content/drive/MyDrive/WeatherPaper/checkpoints/last.pth'
    
    if len(sys.argv) > 1:
        check_weights(sys.argv[1])
    else:
        # Check both last and best if running without args
        print("--- Checking last.pth ---")
        check_weights(default_path)
        
        print("\n--- Checking best_model.pth ---")
        best_path = default_path.replace('last.pth', 'best_model.pth')
        check_weights(best_path)
