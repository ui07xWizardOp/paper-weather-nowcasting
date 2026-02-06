"""
Local Training Script for Weather Nowcasting
=============================================
Run this locally on a machine with GPU to train the model.

Usage:
    python train_local.py

Requirements:
    - PyTorch with CUDA support
    - ~16GB GPU memory recommended (reduce BATCH_SIZE if needed)
    - Preprocessed data in data/batched/
"""

import os
import gc
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "batched")
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

# Training parameters
BATCH_SIZE = 16      # Reduce if running out of GPU memory
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64

# Sequence parameters
T_IN = 24   # 24 hours input
T_OUT = 6   # 6 hours forecast
C, H, W = 2, 31, 41  # Channels (tp, t2m), Height, Width

# ============================================================================
# DATASET
# ============================================================================
class BatchedWeatherDataset(IterableDataset):
    """Memory-efficient dataset that loads batches one at a time."""
    
    def __init__(self, data_dir, split, shuffle=False):
        self.data_dir = os.path.join(data_dir, split)
        self.shuffle = shuffle
        
        # Get batch files
        self.x_files = sorted(glob.glob(os.path.join(self.data_dir, "X_batch_*.npy")))
        self.y_files = sorted(glob.glob(os.path.join(self.data_dir, "Y_batch_*.npy")))
        
        if not self.x_files:
            raise FileNotFoundError(f"No batch files in {self.data_dir}")
        
        # Load metadata
        meta = np.load(os.path.join(self.data_dir, "metadata.npz"))
        self.n_samples = int(meta['n_samples'])
        self.n_batches = int(meta['n_batches'])
        
        print(f"  {split}: {self.n_samples} samples in {self.n_batches} batches")
    
    def __len__(self):
        return self.n_samples
    
    def __iter__(self):
        batch_indices = list(range(len(self.x_files)))
        if self.shuffle:
            np.random.shuffle(batch_indices)
        
        for batch_idx in batch_indices:
            X = np.load(self.x_files[batch_idx])
            Y = np.load(self.y_files[batch_idx])
            
            indices = list(range(len(X)))
            if self.shuffle:
                np.random.shuffle(indices)
            
            for i in indices:
                # (T, H, W, C) -> (T, C, H, W)
                x = torch.tensor(X[i], dtype=torch.float32).permute(0, 3, 1, 2)
                y = torch.tensor(Y[i], dtype=torch.float32).permute(0, 3, 1, 2)
                yield x, y


# ============================================================================
# MODEL
# ============================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size, padding=padding, bias=True
        )
    
    def forward(self, x, hidden):
        h, c = hidden
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    
    def init_hidden(self, batch_size, h, w, device):
        return (torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
                torch.zeros(batch_size, self.hidden_dim, h, w, device=device))


class WeatherNowcaster(nn.Module):
    def __init__(self, in_ch, hidden_dim, out_ch, n_layers=2, kernel_size=3):
        super().__init__()
        self.n_layers = n_layers
        
        # Encoder
        self.encoder = nn.ModuleList([
            ConvLSTMCell(in_ch if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(n_layers)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            ConvLSTMCell(out_ch if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(n_layers)
        ])
        
        self.out_conv = nn.Conv2d(hidden_dim, out_ch, 1)
    
    def forward(self, x, future_steps):
        B, T, C, H, W = x.shape
        device = x.device
        
        # Encode
        hidden = [cell.init_hidden(B, H, W, device) for cell in self.encoder]
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.encoder):
                h, c = cell(inp, hidden[i])
                hidden[i] = (h, c)
                inp = h
        
        # Decode
        dec_hidden = [(h.clone(), c.clone()) for h, c in hidden]
        outputs = []
        dec_in = self.out_conv(dec_hidden[-1][0])
        
        for _ in range(future_steps):
            inp = dec_in
            for i, cell in enumerate(self.decoder):
                h, c = cell(inp, dec_hidden[i])
                dec_hidden[i] = (h, c)
                inp = h
            dec_in = self.out_conv(h)
            outputs.append(dec_in)
        
        return torch.stack(outputs, dim=1)


# ============================================================================
# TRAINING
# ============================================================================
def train():
    print("=" * 60)
    print("LOCAL TRAINING FOR WEATHER NOWCASTING")
    print("=" * 60)
    
    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected. Training will be slow!")
    
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load stats
    print("\n[1/4] Loading data...")
    stats = np.load(os.path.join(DATA_DIR, "stats.npz"), allow_pickle=True)
    mean, std = stats['mean'], stats['std']
    variables = list(stats['variables'])
    print(f"Variables: {variables}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    # Create datasets
    train_dataset = BatchedWeatherDataset(DATA_DIR, 'train', shuffle=True)
    val_dataset = BatchedWeatherDataset(DATA_DIR, 'val', shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    print("\n[2/4] Creating model...")
    model = WeatherNowcaster(C, HIDDEN_DIM, C, n_layers=2, kernel_size=3).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    with torch.no_grad():
        test_in = torch.randn(2, T_IN, C, H, W).to(device)
        test_out = model(test_in, T_OUT)
        print(f"Forward test: {test_in.shape} -> {test_out.shape}")
        del test_in, test_out
        torch.cuda.empty_cache()
    
    # Training setup
    print("\n[3/4] Training setup...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    # Training loop
    print(f"\n[4/4] Training for {NUM_EPOCHS} epochs...")
    print("-" * 60)
    
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x, T_OUT)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
            
            if (batch_idx + 1) % 500 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.6f}")
        
        train_loss /= n_batches
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x, T_OUT)
                val_loss += criterion(out, y).item()
                n_val += 1
        
        val_loss /= n_val
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"  ★ Saved best model")
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, os.path.join(CHECKPOINT_DIR, f'checkpoint_e{epoch+1}.pth'))
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save final model
    print("\n" + "-" * 60)
    print("Saving final model...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'in_ch': C, 'hidden_dim': HIDDEN_DIM, 'out_ch': C,
            'n_layers': 2, 'kernel_size': 3, 'T_IN': T_IN, 'T_OUT': T_OUT
        },
        'mean': mean, 'std': std, 'variables': variables,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses, 'val_losses': val_losses
    }, os.path.join(CHECKPOINT_DIR, 'final_model.pth'))
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, 'training_curve.png'), dpi=150)
    plt.close()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved: {os.path.join(CHECKPOINT_DIR, 'final_model.pth')}")
    print(f"Training curve: {os.path.join(FIGURES_DIR, 'training_curve.png')}")


if __name__ == "__main__":
    train()
