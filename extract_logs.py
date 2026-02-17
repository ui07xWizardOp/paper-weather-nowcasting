
import torch
import csv
import sys
from pathlib import Path

def extract_logs():
    checkpoint_path = Path(r"c:\Users\KIIT0001\Desktop\projects\Paper Weather\results\v4\last.pth")
    output_path = Path(r"c:\Users\KIIT0001\Desktop\projects\Paper Weather\results\v4\training_log.csv")

    if not checkpoint_path.exists():
        print(f"CRITICAL: Checkpoint not found at {checkpoint_path}")
        # Try best_model.pth if last.pth doesn't exist
        checkpoint_path = Path(r"c:\Users\KIIT0001\Desktop\projects\Paper Weather\results\v4\best_model.pth")
        if not checkpoint_path.exists():
             print(f"CRITICAL: best_model.pth also not found.")
             return

    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        train_losses = ckpt.get('train_losses', [])
        val_losses = ckpt.get('val_losses', [])
        epoch = ckpt.get('epoch', len(train_losses)-1)

        print(f"Found {len(train_losses)} epochs of train data and {len(val_losses)} epochs of val data.")

        # Ensure lengths match
        min_len = min(len(train_losses), len(val_losses))
        
        if min_len == 0:
            print("WARNING: No loss data found in checkpoint.")
            return

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'train_loss', 'val_loss'])
            for i in range(min_len):
                writer.writerow([i+1, train_losses[i], val_losses[i]])

        print(f"SUCCESS: Logs extracted to {output_path}")
        
        # Print last few lines
        print("Last 5 epochs:")
        for i in range(max(0, min_len-5), min_len):
            print(f"Epoch {i+1}: Train={train_losses[i]:.4f}, Val={val_losses[i]:.4f}")

    except Exception as e:
        print(f"FAILURE: Could not extract logs. Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_logs()
