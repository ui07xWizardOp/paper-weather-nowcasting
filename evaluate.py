
"""
Evaluation Script for Weather Nowcasting
========================================
Loads a trained model and generates predictions vs ground truth visualizations.
Usage:
    python evaluate.py

Requirements:
    - Trained model checkpoint in 'checkpoints/best_model.pth' (Download from Drive!)
    - Batched test data in 'data/batched/test/'
"""

import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === CONFIGURATION ===
CHECKPOINT_PATH = "checkpoints/best_model.pth"
DATA_DIR = "data/batched"
FIGURES_DIR = "figures/evaluation"
BATCH_SIZE = 4
T_IN = 24
T_OUT = 6
VARIABLES = ['Total Precip (tp)', 'Temp (t2m)']

# Ensure directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# === MODEL DEFINITION (Must match training!) ===
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=kernel_size//2)
    def forward(self, x, hidden):
        h, c = hidden
        gates = self.conv(torch.cat([x, h], dim=1))
        # Split into 4 chunks: i, f, o, g
        chunks = torch.chunk(gates, 4, dim=1)
        i, f, o, g = chunks[0], chunks[1], chunks[2], chunks[3]
        
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next
    def init_hidden(self, B, H, W, dev):
        return (torch.zeros(B, self.hidden_dim, H, W, device=dev),
                torch.zeros(B, self.hidden_dim, H, W, device=dev))

class WeatherNowcaster(nn.Module):
    def __init__(self, in_ch, hidden_dim, out_ch, n_layers=2):
        super().__init__()
        self.encoder = nn.ModuleList([ConvLSTMCell(in_ch if i==0 else hidden_dim, hidden_dim, 3) for i in range(n_layers)])
        self.decoder = nn.ModuleList([ConvLSTMCell(out_ch if i==0 else hidden_dim, hidden_dim, 3) for i in range(n_layers)])
        self.out_conv = nn.Conv2d(hidden_dim, out_ch, 3, padding=1)  # Must match training
    
    def forward(self, x, future_steps):
        B, T, C, H, W = x.shape
        hidden = [cell.init_hidden(B, H, W, x.device) for cell in self.encoder]
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.encoder):
                h, c = cell(inp, hidden[i])
                hidden[i] = (h, c)
                inp = h
        
        dec_hidden = [(h.clone(), c.clone()) for h, c in hidden]
        outputs = []
        # Initial decoder input is the last encoder output (mapped to out_ch)
        dec_in = self.out_conv(dec_hidden[-1][0])
        
        for _ in range(future_steps):
            for i, cell in enumerate(self.decoder):
                h, c = cell(dec_in if i==0 else h, dec_hidden[i])
                dec_hidden[i] = (h, c)
            dec_in = self.out_conv(h)
            outputs.append(dec_in)
        return torch.stack(outputs, dim=1)

# === UTILITIES ===
def load_model(device):
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("   Please download 'best_model.pth' from your Google Drive (checkpoints folder).")
        return None
        
    model = WeatherNowcaster(in_ch=2, hidden_dim=128, out_ch=2).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # Handle both full checkpoint dict and raw state_dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"✓ Loaded checkpoint (Val Loss: {checkpoint.get('val_loss', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded state dict")
    
    model.eval()
    return model

def get_test_batch():
    # Helper to get one batch of test data
    files = sorted(glob.glob(f"{DATA_DIR}/test/X_batch_*.npy"))
    y_files = sorted(glob.glob(f"{DATA_DIR}/test/Y_batch_*.npy"))
    
    if not files:
        print("❌ No test data found in data/batched/test/")
        return None, None

    # Load random file
    idx = np.random.randint(0, len(files))
    # Memory-mapped loading — avoids loading entire file into RAM
    X = np.load(files[idx], mmap_mode='r')  # (N, T, H, W, C)
    Y = np.load(y_files[idx], mmap_mode='r')
    
    # Select random batch
    n_samples = len(X)
    indices = np.random.choice(n_samples, BATCH_SIZE, replace=False)
    
    # Convert to torch: (B, T, H, W, C) -> (B, T, C, H, W)
    x_batch = torch.from_numpy(X[indices]).float().permute(0, 1, 4, 2, 3)
    y_batch = torch.from_numpy(Y[indices]).float().permute(0, 1, 4, 2, 3)
    
    return x_batch, y_batch

def visualize_prediction(x, y_true, y_pred, sample_idx=0):
    # Determine device (cpu/cuda) for moving tensors to numpy
    # x, y_true, y_pred are tensors
    
    # Select variable to visualize (0=Precip, 1=Temp)
    var_idx = 0 
    var_name = VARIABLES[var_idx]
    
    # Get sequences for the sample: (T, H, W)
    # Input history
    hist = x[sample_idx, :, var_idx].cpu().numpy()
    # Ground truth future
    true_fut = y_true[sample_idx, :, var_idx].cpu().numpy()
    # Predicted future
    pred_fut = y_pred[sample_idx, :, var_idx].cpu().numpy()
    
    # --- 1. Static Plot (Last input -> Last target vs Last pred) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Normalize/Clip for visualization usually good to keep as is if standardized
    # but let's assume raw values are standardized.
    vmin, vmax = -2, 2 # Approx range for standardized data
    
    axes[0].imshow(hist[-1], cmap='Blues', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Last Input (T={T_IN})")
    
    axes[1].imshow(true_fut[-1], cmap='Blues', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Ground Truth (T={T_IN+T_OUT})")
    
    axes[2].imshow(pred_fut[-1], cmap='Blues', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Prediction (T={T_IN+T_OUT})")
    
    plt.suptitle(f"Weather Prediction: {var_name}")
    save_path = f"{FIGURES_DIR}/prediction_sample_{sample_idx}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved plot: {save_path}")

def main():
    print("=== Paper Weather Evaluation ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Model
    model = load_model(device)
    if model is None: return

    # 2. Get Data
    x, y = get_test_batch()
    if x is None: return
    
    x = x.to(device)
    y = y.to(device)
    
    # 3. Predict
    print("Running inference...")
    with torch.no_grad():
        start = time.time()
        preds = model(x, T_OUT) # (B, T_out, C, H, W)
        print(f"Inference time: {time.time()-start:.3f}s")
        
    # 4. Calculate Loss
    mse = nn.MSELoss()(preds, y).item()
    print(f"Test Batch MSE: {mse:.6f}")
    
    # 5. Visualize
    print("Generating visualizations...")
    for i in range(BATCH_SIZE):
        visualize_prediction(x, y, preds, sample_idx=i)
        
    print("\n✓ Evaluation Complete. Check 'figures/evaluation' folder.")

if __name__ == "__main__":
    main()
