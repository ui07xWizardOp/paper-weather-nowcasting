
"""
CPU Training / Debug Script for Weather Nowcasting
==================================================
Usage:
    python train_cpu.py

Configured for:
- CPU execution (no GPU required)
- Fast debugging (tiny model, 1 epoch, few batches)
"""

import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# === FAST DEBUG CONFIG ===
DEBUG_MODE = True  # Forces 1 epoch, few batches
BATCH_SIZE = 4     # Low batch size for CPU
HIDDEN_DIM = 16    # Tiny model for speed
NUM_EPOCHS = 1     # Just to verify pipeline
T_IN, T_OUT = 24, 6
C, H, W = 2, 31, 41
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "batched")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class BatchedWeatherDataset(IterableDataset):
    def __init__(self, data_dir, split, limit_files=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, split, "X_batch_*.npy")))
        self.y_files = sorted(glob.glob(os.path.join(data_dir, split, "Y_batch_*.npy")))
        if limit_files:
            self.files = self.files[:limit_files]
            self.y_files = self.y_files[:limit_files]
        print(f"[{split}] Loaded {len(self.files)} batch files")
    
    def __iter__(self):
        for xf, yf in zip(self.files, self.y_files):
            # Memory-mapped loading — OS caches in available RAM
            X = np.load(xf, mmap_mode='r')
            Y = np.load(yf, mmap_mode='r')
            indices = np.arange(len(X))
            if not DEBUG_MODE: np.random.shuffle(indices)
            for i in indices:
                # Copy from mmap to contiguous array, then to tensor
                # Convert (T, H, W, C) -> (T, C, H, W)
                x = torch.from_numpy(np.array(X[i])).float().permute(0, 3, 1, 2)
                y = torch.from_numpy(np.array(Y[i])).float().permute(0, 3, 1, 2)
                yield x, y

class ConvLSTMCell(nn.Module):
    def __init__(self, in_c, hid_c, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_c + hid_c, 4 * hid_c, k, padding=k//2)
        self.hid_c = hid_c
    
    def forward(self, x, h_prev):
        h, c = h_prev
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next
    
    def init_hidden(self, b, h, w):
        return (torch.zeros(b, self.hid_c, h, w), torch.zeros(b, self.hid_c, h, w))

class WeatherModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified 1-layer for speed
        self.enc = ConvLSTMCell(C, HIDDEN_DIM)
        self.dec = ConvLSTMCell(C, HIDDEN_DIM)
        self.out = nn.Conv2d(HIDDEN_DIM, C, 1)

    def forward(self, x, steps):
        b, t, _, h, w = x.shape
        # Encode
        hidden = self.enc.init_hidden(b, h, w)
        for i in range(t):
            hidden = self.enc(x[:, i], hidden)
        
        # Decode
        outputs = []
        dec_in = self.out(hidden[0]) # Start with last state
        dec_hid = hidden
        
        for _ in range(steps):
            dec_hid = self.dec(dec_in, dec_hid)
            dec_in = self.out(dec_hid[0])
            outputs.append(dec_in)
            
        return torch.stack(outputs, dim=1)

def main():
    print("=== CPU DRY RUN START ===")
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        print(f"System RAM: {mem.total / 1e9:.1f} GB, Available: {mem.available / 1e9:.1f} GB")
    
    # 1. Load Data (Limit to 2 files for speed)
    train_ds = BatchedWeatherDataset(DATA_DIR, 'train', limit_files=2)
    val_ds = BatchedWeatherDataset(DATA_DIR, 'val', limit_files=1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
    
    # 2. Model
    model = WeatherModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    print(f"Model created. Params: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Train Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        model.train()
        losses = []
        start = time.time()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x, T_OUT)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if i % 10 == 0:
                print(f"  Batch {i}: Loss {loss.item():.4f}")
            if i >= 20: break # Stop quickly
            
        print(f"  Train Time: {time.time()-start:.1f}s | Avg Loss: {np.mean(losses):.4f}")
        
        # Save
        path = os.path.join(CHECKPOINT_DIR, "debug_model.pth")
        torch.save(model.state_dict(), path)
        print(f"  Saved: {path}")

    print("\n=== SUCCESS: Pipeline verified on CPU! ===")

if __name__ == "__main__":
    main()
