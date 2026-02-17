
import json
import re

INPUT_NB = 'notebooks/03_training_resume.ipynb'
OUTPUT_NB = 'notebooks/03_training_optimized.ipynb'

# --- NEW CONTENT FOR CELLS ---

cell_5_data_loading = [
    "# Cell 5: Optimized Data Loading (Parallel)\n",
    "import torch\n",
    "from torch.utils.data import IterableDataset, DataLoader\n",
    "\n",
    "BATCH_SIZE = 32  # Reduced to avoid OOM on T4 (15GB)\n",
    "NUM_WORKERS = 2  # Parallel workers (Reduced for Colab T4 stability)\n",
    "import os\n",
    "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
    "\n",
    "def get_batch_files(split):\n",
    "    split_dir = f'{BATCHED_DIR}/{split}'\n",
    "    x_files = sorted(glob.glob(f'{split_dir}/X_batch_*.npy'))\n",
    "    y_files = sorted(glob.glob(f'{split_dir}/Y_batch_*.npy'))\n",
    "    return x_files, y_files\n",
    "\n",
    "class BatchedWeatherDataset(IterableDataset):\n",
    "    def __init__(self, split, shuffle=False):\n",
    "        self.split = split\n",
    "        self.shuffle = shuffle\n",
    "        self.x_files, self.y_files = get_batch_files(split)\n",
    "        \n",
    "    def __iter__(self):\n",
    "        worker_info = torch.utils.data.get_worker_info()\n",
    "        x_files = list(self.x_files)\n",
    "        y_files = list(self.y_files)\n",
    "        \n",
    "        # Distribute files among workers\n",
    "        if worker_info is not None:\n",
    "            per_worker = int(np.ceil(len(x_files) / float(worker_info.num_workers)))\n",
    "            iter_start = worker_info.id * per_worker\n",
    "            iter_end = min(iter_start + per_worker, len(x_files))\n",
    "            x_files = x_files[iter_start:iter_end]\n",
    "            y_files = y_files[iter_start:iter_end]\n",
    "            \n",
    "        # Shuffle files\n",
    "        file_indices = list(range(len(x_files)))\n",
    "        if self.shuffle:\n",
    "            np.random.shuffle(file_indices)\n",
    "            \n",
    "        for idx in file_indices:\n",
    "            try:\n",
    "                X = np.load(x_files[idx], mmap_mode='r')\n",
    "                Y = np.load(y_files[idx], mmap_mode='r')\n",
    "                n = len(X)\n",
    "                indices = np.random.permutation(n) if self.shuffle else np.arange(n)\n",
    "                \n",
    "                # Yield batches\n",
    "                for start in range(0, n, BATCH_SIZE):\n",
    "                    batch_idx = indices[start:start+BATCH_SIZE]\n",
    "                    # Drop incomplete batches during training\n",
    "                    if len(batch_idx) < BATCH_SIZE and self.split == 'train': continue\n",
    "                    \n",
    "                    # Copy from mmap to contiguous tensor\n",
    "                    x = torch.from_numpy(np.array(X[batch_idx])).float().permute(0, 1, 4, 2, 3)\n",
    "                    y = torch.from_numpy(np.array(Y[batch_idx])).float().permute(0, 1, 4, 2, 3)\n",
    "                    yield x, y\n",
    "            except Exception as e:\n",
    "                print(f'Error loading {x_files[idx]}: {e}')\n",
    "\n",
    "def get_loader(split, shuffle=False):\n",
    "    ds = BatchedWeatherDataset(split, shuffle=shuffle)\n",
    "    # batch_size=None because dataset yields batches\n",
    "    return DataLoader(ds, batch_size=None, num_workers=NUM_WORKERS,\n",
    "                      pin_memory=True, prefetch_factor=4, persistent_workers=True)\n",
    "\n",
    "# RAM diagnostics\n",
    "import psutil\n",
    "mem = psutil.virtual_memory()\n",
    "print(f'System RAM: {mem.total / 1e9:.1f} GB, Available: {mem.available / 1e9:.1f} GB')\n",
    "\n",
    "# Load stats for reference\n",
    "stats = np.load(f'{BATCHED_DIR}/stats.npz', allow_pickle=True)\n",
    "mean, std = stats['mean'], stats['std']\n",
    "variables = list(stats['variables'])\n",
    "print(f'Variables: {variables}')\n",
    "print(f'Norm stats -- Mean: {mean}, Std: {std}')\n",
    "print(f'Mean range: [{mean.min():.4f}, {mean.max():.4f}]')\n",
    "print(f'Std range:  [{std.min():.4f}, {std.max():.4f}]')\n",
    "\n",
    "# Batch counts (Approx)\n",
    "train_files, _ = get_batch_files('train')\n",
    "val_files, _ = get_batch_files('val')\n",
    "test_files, _ = get_batch_files('test')\n",
    "\n",
    "SAMPLES_PER_FILE = 500\n",
    "BATCHES_PER_FILE = (SAMPLES_PER_FILE + BATCH_SIZE - 1) // BATCH_SIZE\n",
    "n_train_batches = len(train_files) * BATCHES_PER_FILE\n",
    "n_val_batches = len(val_files) * BATCHES_PER_FILE\n",
    "n_test_batches = len(test_files) * BATCHES_PER_FILE\n",
    "\n",
    "print(f'Train batches (approx): {n_train_batches}')\n",
    "print(f'Val batches (approx): {n_val_batches}')\n"
]

def main():
    with open(INPUT_NB, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 1. Update Cell 5: Data Loading
    # Find cell with "Data Loading Utilities"
    cell_5_idx = -1
    for i, cell in enumerate(nb['cells']):
        if "Data Loading Utilities" in "".join(cell.get('source', [])):
            cell_5_idx = i
            break
            
    if cell_5_idx != -1:
        nb['cells'][cell_5_idx]['source'] = cell_5_data_loading
        print(f"Updated Data Loading in Cell {cell_5_idx}")
    else:
        print("Warning: Could not find Data Loading Utilities cell")

    # 1.2 Global Fix: Replace deprecated autocast in ALL cells
    print("Applying global fix for deprecated autocast...")
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell.get('source', []))
            if "with autocast():" in src:
                new_src = src.replace("with autocast():", "with torch.amp.autocast('cuda'):")
                cell['source'] = new_src.splitlines(True)

    # 1.3 Global Fix: Replace batch_generator with get_loader in ALL cells
    print("Applying global fix for batch_generator references...")
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell.get('source', []))
            if "batch_generator(" in src:
                new_src = src.replace("batch_generator('val')", "get_loader('val')")
                new_src = new_src.replace("batch_generator('test', BATCH_SIZE)", "get_loader('test')")
                new_src = new_src.replace("batch_generator('train')", "get_loader('train', shuffle=True)")
                new_src = new_src.replace("batch_generator('test')", "get_loader('test')")
                cell['source'] = new_src.splitlines(True)

    # 1.4 Global Fix: Replace 1x1 output conv with 3x3 (reduces checker artifacts)
    print("Applying global fix for output conv (1x1 -> 3x3)...")
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell.get('source', []))
            if "self.out_conv = nn.Conv2d(hidden_dim, out_ch, 1)" in src:
                new_src = src.replace(
                    "self.out_conv = nn.Conv2d(hidden_dim, out_ch, 1)",
                    "self.out_conv = nn.Conv2d(hidden_dim, out_ch, 3, padding=1)  # 3x3 for spatial smoothing"
                )
                cell['source'] = new_src.splitlines(True)
                print("  Updated output conv to 3x3")

    # 0.5 Update Config Cell
    # 0.5 Update Config Cell
    print("Updating Config Cell...")
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        new_src = src
        # Fix Stride
        if "STRIDE = 6" in src:
             new_src = new_src.replace("STRIDE = 6", "STRIDE = 12")
             print("  Updated STRIDE to 12 in Config Cell")
        
        # Force data path update (Independent of Stride)
        if "DATA_DIR =" in src:
             new_src = re.sub(r"DATA_DIR = .*", "DATA_DIR = os.path.join(base_path, 'data', 'batched_v4_linear')", new_src)
             print("  Updated DATA_DIR to v4_linear in Config Cell")
             
        if new_src != src:
             nb['cells'][i]['source'] = new_src.splitlines(True)

    # ... (existing Preprocessing / Data Copy / Auto-Resume logic) ...

    # Duplicate Benchmark Update Removed


    # 1.5 Global Fix: Add stride=12 to create_sequences (reduces 95.8% overlap to 50%)
    print("Applying global fix for sequence stride...")
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell.get('source', []))
            if "def create_sequences(data, t_in, t_out):" in src and "for i in range(t_in, n - t_out + 1):" in src:
                new_src = src.replace(
                    "for i in range(t_in, n - t_out + 1):",
                    "# Use STRIDE from Config\n        for i in range(t_in, n - t_out + 1, STRIDE):"
                )
                cell['source'] = new_src.splitlines(True)
                print("  Updated create_sequences to use global STRIDE")

    # DATA CLEANUP: Remove log1p if present (Handle contaminated input)
    print("Ensuring log1p is REMOVED...")
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell.get('source', []))
            if "np.log1p" in src and "data[..., 0]" in src:
                # remove log1p lines
                new_src = src.replace("            # Log1p transform precipitation (channel 0) to compress heavy tail\n", "")
                new_src = new_src.replace("            data[..., 0] = np.log1p(np.maximum(data[..., 0], 0))\n", "")
                # remove stats log1p
                new_src = new_src.replace("                    # Log1p transform precipitation before computing stats\n", "")
                new_src = new_src.replace("                    data[..., 0] = np.log1p(np.maximum(data[..., 0], 0))\n", "")
                
                if new_src != src:
                    cell['source'] = new_src.splitlines(True)
                    print("  removed log1p traces from cell")

    # GLOBAL FIX: Ensure n_layers=4 (Fixing 3.5M parameter issue)
    print("Ensuring n_layers=4 everywhere...")
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell.get('source', []))
            if "n_layers=2" in src:
                new_src = src.replace("n_layers=2", "n_layers=4")
                cell['source'] = new_src.splitlines(True)
                print("  Updated n_layers=2 -> 4")

    # 1.7 Inject Data Copy Cell (Critical for Speed)
    resume_idx = -1
    for i, cell in enumerate(nb['cells']):
        if "Auto-Resume Logic" in "".join(cell.get('source', [])):
            resume_idx = i
            break

    data_copy_code = [
        "# Cell: High-Speed Data caching (Drive -> Local)\n",
        "# Copying data to local VM speed up training by 10x\n",
        "import shutil\n",
        "import os\n",
        "\n",
        "DRIVE_DATA = '/content/drive/MyDrive/WeatherPaper/data/batched_v4_linear'\n",
        "LOCAL_DATA = '/content/data/batched_v4_linear'\n",
        "# Fallback text path from Cell 1 setup\n",
        "DEFAULT_DATA = '/content/weather_nowcasting/data/batched_v4_linear'\n",
        "\n",
        "if not os.path.exists(LOCAL_DATA) and os.path.exists(DRIVE_DATA):\n",
        "    print(f'⏳ Copying data from Drive to Local Disk... (This takes ~2-3 mins)')\n",
        "    print(f'   Source: {DRIVE_DATA}')\n",
        "    print(f'   Dest:   {LOCAL_DATA}')\n",
        "    try:\n",
        "        shutil.copytree(DRIVE_DATA, LOCAL_DATA)\n",
        "        print('✓ Data copied! Training will be fast now.')\n",
        "        BATCHED_DIR = LOCAL_DATA\n",
        "    except Exception as e:\n",
        "        print(f'⚠️ Copy failed: {e}. Using default path.')\n",
        "        BATCHED_DIR = DEFAULT_DATA\n",
        "elif os.path.exists(LOCAL_DATA):\n",
        "    print('✓ Local data already exists.')\n",
        "    BATCHED_DIR = LOCAL_DATA\n",
        "elif os.path.exists(DEFAULT_DATA):\n",
        "    print('✓ Using generated data in workspace.')\n",
        "    BATCHED_DIR = DEFAULT_DATA\n",
        "else:\n",
        "    print(f'⚠️ Warning: Data not found at {DRIVE_DATA} or {LOCAL_DATA}')\n",
        "    print('   Please ensure you uploaded data/batched to Drive/WeatherPaper/data/batched')\n",
        "    BATCHED_DIR = DEFAULT_DATA # Attempt to continue\n",
        "\n",
        "print(f'Data Source: {BATCHED_DIR}')\n"
    ]

    if resume_idx != -1:
        # Check if already inserted
        next_cell = nb['cells'][resume_idx + 1] if resume_idx + 1 < len(nb['cells']) else {}
        if "High-Speed Data caching" in "".join(next_cell.get('source', [])):
             print("Data Copy cell already present, updating...")
             nb['cells'][resume_idx + 1]['source'] = data_copy_code
        else:
             new_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": data_copy_code
             }
             nb['cells'].insert(resume_idx + 1, new_cell)
             print(f"Injected Data Copy Logic after cell {resume_idx}")

    # 1.8 Replace Auto-Resume Logic (Robust against shape mismatch)
    print("Updating Auto-Resume Logic...")
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        if "Auto-Resume Logic" in src:
            nb['cells'][i]['source'] = [
                "# Cell: Auto-Resume Logic & Drive Mounting\n",
                "import os\n",
                "import torch\n",
                "try:\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive')\n",
                "    print('\u2713 Google Drive mounted')\n",
                "    CHECKPOINT_DIR = '/content/drive/MyDrive/WeatherPaper/checkpoints'\n",
                "    os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
                "    print(f'\u2713 Checkpoints will be saved to: {CHECKPOINT_DIR}')\n",
                "except ImportError:\n",
                "    print('\u26a0\ufe0f Not running in Colab, using local paths')\n",
                "    CHECKPOINT_DIR = 'checkpoints'\n",
                "\n",
                "train_losses, val_losses = [], []\n",
                "RESUME_FROM = None\n",
                "start_epoch = 0\n",
                "\n",
                "# Check for existing LAST checkpoint\n",
                "if os.path.exists(f'{CHECKPOINT_DIR}/last.pth'):\n",
                "    RESUME_FROM = f'{CHECKPOINT_DIR}/last.pth'\n",
                "elif os.path.exists(f'{CHECKPOINT_DIR}/best_model.pth'):\n",
                "    RESUME_FROM = f'{CHECKPOINT_DIR}/best_model.pth'\n",
                "\n",
                "if RESUME_FROM and os.path.exists(RESUME_FROM):\n",
                "    print(f'Loading checkpoint: {RESUME_FROM}')\n",
                "    try:\n",
                "        checkpoint = torch.load(RESUME_FROM, map_location='cuda' if torch.cuda.is_available() else 'cpu')\n",
                "        if 'model' in checkpoint:\n",
                "            model.load_state_dict(checkpoint['model'])\n",
                "        else:\n",
                "            model.load_state_dict(checkpoint)\n",
                "        \n",
                "        if 'train_losses' in checkpoint:\n",
                "            train_losses = checkpoint['train_losses']\n",
                "            val_losses = checkpoint['val_losses']\n",
                "        if 'epoch' in checkpoint:\n",
                "            start_epoch = checkpoint['epoch'] + 1\n",
                "            print(f'Resuming from epoch {start_epoch}')\n",
                "        else:\n",
                "            print('Warning: Epoch info not found, fine-tuning from 0')\n",
                "    except RuntimeError as e:\n",
                "        print(f'\\n\u26a0\ufe0f Checkpoint mismatch (expected due to architecture change): {e}')\n",
                "        print('\u27f3 Starting fresh training from epoch 0...')\n",
                "        # Re-init model to be safe\n",
                "        model = WeatherNowcaster(IN_CHANNELS, HIDDEN_DIM, OUT_CHANNELS, n_layers=4).to(device)\n",
                "        model.apply(init_weights)  # Re-apply Kaiming init\n",
                "        start_epoch = 0\n"
            ]
            print(f"  Updated Auto-Resume Logic at index {i}")
            break

    # 2. Update Cell 6: Benchmark & Model Def
    cell_6_idx = -1
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        if "Benchmarking model" in src:  # Removed 'batch_generator' check as it might be replaced by global fix
            cell_6_idx = i
            # Replace batch_generator call with loader (if not already done)
            new_src = src.replace("batch_generator('train')", "get_loader('train', shuffle=True)")
            # Append init_weights definition here so it's available for Resume Logic
            # Check if init_weights is already present to avoid duplication
            if "def init_weights(m):" not in new_src:
                 new_src += "\n# Proper weight initialization\ndef init_weights(m):\n    if isinstance(m, nn.Conv2d):\n        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')\n        if m.bias is not None:\n            nn.init.zeros_(m.bias)\n"
            nb['cells'][i]['source'] = new_src.splitlines(True)
            print("  Updated Cell 6 (Benchmark + Model init_weights)")

    # 3. Replace Training Loop (Cell 7)
    cell_loop_idx = -1
    for i, cell in enumerate(nb['cells']):
        if "Cell 7: Training Loop" in "".join(cell.get('source', [])):
            cell_loop_idx = i
            
            new_lines = []
            new_lines.append("# Cell 7: Training Loop with AMP + Weighted Loss\n")
            new_lines.append("import torch.nn.functional as F\n")
            new_lines.append("\n")
            new_lines.append("# WeightedMSE: upweight non-zero precipitation cells\n")
            new_lines.append("class WeightedMSE(nn.Module):\n")
            new_lines.append("    def __init__(self, zero_weight=1.0, nonzero_weight=10.0, threshold=0.1):\n")
            new_lines.append("        super().__init__()\n")
            new_lines.append("        self.zero_weight = zero_weight\n")
            new_lines.append("        self.nonzero_weight = nonzero_weight\n")
            new_lines.append("        self.threshold = threshold\n")
            new_lines.append("    def forward(self, pred, target):\n")
            new_lines.append("        weight = torch.where(target.abs() > self.threshold,\n")
            new_lines.append("                             self.nonzero_weight, self.zero_weight)\n")
            new_lines.append("        return (weight * (pred - target) ** 2).mean()\n")
            new_lines.append("\n")
            new_lines.append("# SSIMLoss with corrected constants for z-scored data\n")

    # ROGUE BLOCK DELETED

    # ROGUE STATS BLOCK DELETED

    # 1.7 Inject Data Copy Cell (Critical for Speed)
    # Find cell with "Auto-Resume Logic" to insert after it
    resume_idx = -1
    for i, cell in enumerate(nb['cells']):
        if "Auto-Resume Logic" in "".join(cell.get('source', [])):
            resume_idx = i
            break
    
    data_copy_code = [
        "# Cell: High-Speed Data caching (Drive -> Local)\n",
        "# Copying data to local VM speed up training by 10x\n",
        "import shutil\n",
        "import os\n",
        "\n",
        "DRIVE_DATA = '/content/drive/MyDrive/WeatherPaper/data/batched'\n",
        "LOCAL_DATA = '/content/data/batched'\n",
        "# Fallback text path from Cell 1 setup\n",
        "DEFAULT_DATA = '/content/weather_nowcasting/data/batched'\n",
        "\n",
        "if not os.path.exists(LOCAL_DATA) and os.path.exists(DRIVE_DATA):\n",
        "    print(f'⏳ Copying data from Drive to Local Disk... (This takes ~2-3 mins)')\n",
        "    print(f'   Source: {DRIVE_DATA}')\n",
        "    print(f'   Dest:   {LOCAL_DATA}')\n",
        "    try:\n",
        "        shutil.copytree(DRIVE_DATA, LOCAL_DATA)\n",
        "        print('✓ Data copied! Training will be fast now.')\n",
        "        BATCHED_DIR = LOCAL_DATA\n",
        "    except Exception as e:\n",
        "        print(f'⚠️ Copy failed: {e}. Using default path.')\n",
        "        BATCHED_DIR = DEFAULT_DATA\n",
        "elif os.path.exists(LOCAL_DATA):\n",
        "    print('✓ Local data already exists.')\n",
        "    BATCHED_DIR = LOCAL_DATA\n",
        "elif os.path.exists(DEFAULT_DATA):\n",
        "    print('✓ Using generated data in workspace.')\n",
        "    BATCHED_DIR = DEFAULT_DATA\n",
        "else:\n",
        "    print(f'⚠️ Warning: Data not found at {DRIVE_DATA} or {LOCAL_DATA}')\n",
        "    print('   Please ensure you uploaded data/batched to Drive/WeatherPaper/data/batched')\n",
        "    BATCHED_DIR = DEFAULT_DATA # Attempt to continue\n",
        "\n",
        "print(f'Data Source: {BATCHED_DIR}')\n"
    ]

    if resume_idx != -1:
        # Check if already inserted
        next_cell = nb['cells'][resume_idx + 1] if resume_idx + 1 < len(nb['cells']) else {}
        if "High-Speed Data caching" in "".join(next_cell.get('source', [])):
             print("Data Copy cell already present, updating...")
             nb['cells'][resume_idx + 1]['source'] = data_copy_code
        else:
             new_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": data_copy_code
             }
             nb['cells'].insert(resume_idx + 1, new_cell)
             print(f"Injected Data Copy Logic after cell {resume_idx}")

    # 1.8 Replace Auto-Resume Logic (Robust against shape mismatch)
    print("Updating Auto-Resume Logic...")
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        if "Auto-Resume Logic" in src:
            nb['cells'][i]['source'] = [
                "# Cell: Auto-Resume Logic & Drive Mounting\n",
                "import os\n",
                "import torch\n",
                "try:\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive')\n",
                "    print('\u2713 Google Drive mounted')\n",
                "    CHECKPOINT_DIR = '/content/drive/MyDrive/WeatherPaper/checkpoints'\n",
                "    os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
                "    print(f'\u2713 Checkpoints will be saved to: {CHECKPOINT_DIR}')\n",
                "except ImportError:\n",
                "    print('\u26a0\ufe0f Not running in Colab, using local paths')\n",
                "    CHECKPOINT_DIR = 'checkpoints'\n",
                "\n",
                "train_losses, val_losses = [], []\n",
                "RESUME_FROM = None\n",
                "start_epoch = 0\n",
                "\n",
                "# Check for existing LAST checkpoint\n",
                "if os.path.exists(f'{CHECKPOINT_DIR}/last.pth'):\n",
                "    RESUME_FROM = f'{CHECKPOINT_DIR}/last.pth'\n",
                "elif os.path.exists(f'{CHECKPOINT_DIR}/best_model.pth'):\n",
                "    RESUME_FROM = f'{CHECKPOINT_DIR}/best_model.pth'\n",
                "\n",
                "if RESUME_FROM and os.path.exists(RESUME_FROM):\n",
                "    print(f'Loading checkpoint: {RESUME_FROM}')\n",
                "    try:\n",
                "        checkpoint = torch.load(RESUME_FROM, map_location='cuda' if torch.cuda.is_available() else 'cpu')\n",
                "        if 'model' in checkpoint:\n",
                "            model.load_state_dict(checkpoint['model'])\n",
                "        else:\n",
                "            model.load_state_dict(checkpoint)\n",
                "        \n",
                "        if 'train_losses' in checkpoint:\n",
                "            train_losses = checkpoint['train_losses']\n",
                "            val_losses = checkpoint['val_losses']\n",
                "        if 'epoch' in checkpoint:\n",
                "            start_epoch = checkpoint['epoch'] + 1\n",
                "            print(f'Resuming from epoch {start_epoch}')\n",
                "        else:\n",
                "            print('Warning: Epoch info not found, fine-tuning from 0')\n",
                "    except RuntimeError as e:\n",
                "        print(f'\\n\u26a0\ufe0f Checkpoint mismatch (expected due to architecture change): {e}')\n",
                "        print('\u27f3 Starting fresh training from epoch 0...')\n",
                "        # Re-init model to be safe\n",
                "        model = WeatherNowcaster(IN_CHANNELS, HIDDEN_DIM, OUT_CHANNELS, n_layers=4).to(device)\n",
                "        model.apply(init_weights)  # Re-apply Kaiming init\n",
                "        start_epoch = 0\n"
            ]
            print(f"  Updated Auto-Resume Logic at index {i}")
            break

    # Duplicate Benchmark Update Removed (Logic moved to top of file)

    # 3. Update Cell 8: Training Loop (It might be index 8 now due to resume cell)
    # Find cell with "Training Loop"
    cell_loop_idx = -1
    for i, cell in enumerate(nb['cells']):
        if "Training Loop" in "".join(cell.get('source', [])):
            cell_loop_idx = i
            src = "".join(cell.get('source', []))
            
            # Use a fresh manual construction for the loop to avoid regex fragility
            new_lines = []
            
            # --- Header / Setup ---
            new_lines.append("# Cell 7: Training Loop with AMP + Weighted Loss\n")
            new_lines.append("import torch.nn.functional as F\n")
            new_lines.append("\n")
            # WeightedMSE: upweight non-zero precipitation cells
            new_lines.append("class WeightedMSE(nn.Module):\n")
            new_lines.append("    def __init__(self, zero_weight=1.0, nonzero_weight=10.0, threshold=0.1):\n")
            new_lines.append("        super().__init__()\n")
            new_lines.append("        self.zero_weight = zero_weight\n")
            new_lines.append("        self.nonzero_weight = nonzero_weight\n")
            new_lines.append("        self.threshold = threshold\n")
            new_lines.append("    def forward(self, pred, target):\n")
            new_lines.append("        weight = torch.where(target.abs() > self.threshold,\n")
            new_lines.append("                             self.nonzero_weight, self.zero_weight)\n")
            new_lines.append("        return (weight * (pred - target) ** 2).mean()\n")
            new_lines.append("\n")
            # SSIMLoss with corrected constants for z-scored data
            new_lines.append("class SSIMLoss(nn.Module):\n")
            new_lines.append("    def __init__(self, window_size=7, data_range=6.0):\n")
            new_lines.append("        super().__init__()\n")
            new_lines.append("        self.window_size = window_size\n")
            new_lines.append("        self.data_range = data_range  # ±3σ for z-scored data\n")
            new_lines.append("    def gaussian_window(self, size, sigma=1.5):\n")
            new_lines.append("        coords = torch.arange(size, dtype=torch.float32) - size // 2\n")
            new_lines.append("        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))\n")
            new_lines.append("        g = g / g.sum()\n")
            new_lines.append("        return g.unsqueeze(1) @ g.unsqueeze(0)\n")
            new_lines.append("    def forward(self, pred, target):\n")
            new_lines.append("        B, T, C, H, W = pred.shape\n")
            new_lines.append("        pred = pred.reshape(B * T, C, H, W)\n")
            new_lines.append("        target = target.reshape(B * T, C, H, W)\n")
            new_lines.append("        window = self.gaussian_window(self.window_size).to(pred.device)\n")
            new_lines.append("        window = window.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)\n")
            new_lines.append("        pad = self.window_size // 2\n")
            new_lines.append("        mu_p = F.conv2d(pred, window, padding=pad, groups=C)\n")
            new_lines.append("        # mu_p duplicate removed\n")
            new_lines.append("        mu_t = F.conv2d(target, window, padding=pad, groups=C)\n")
            new_lines.append("        # Instability Fix: Clamp variance to be non-negative to avoid NaN/-inf in float16\n")
            new_lines.append("        sigma_p2 = F.relu(F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_p ** 2)\n")
            new_lines.append("        sigma_t2 = F.relu(F.conv2d(target * target, window, padding=pad, groups=C) - mu_t ** 2)\n")
            new_lines.append("        sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_p * mu_t\n")
            new_lines.append("        C1 = (0.01 * self.data_range) ** 2\n")
            new_lines.append("        C2 = (0.03 * self.data_range) ** 2\n")
            new_lines.append("        # Add epsilon to denominator for safety\n")
            new_lines.append("        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \\\n")
            new_lines.append("               ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p2 + sigma_t2 + C2) + 1e-8)\n")
            new_lines.append("        return 1 - torch.clamp(ssim, min=-1.0, max=1.0).mean()\n")
            new_lines.append("\n")
            # Combined loss: WeightedMSE + SSIM + MAE
            new_lines.append("wmse_criterion = WeightedMSE(zero_weight=1.0, nonzero_weight=10.0, threshold=0.1)\n")
            new_lines.append("ssim_criterion = SSIMLoss(window_size=7, data_range=6.0)\n")
            new_lines.append("mae_criterion = nn.L1Loss()\n")
            new_lines.append("def criterion(pred, target):\n")
            new_lines.append("    # Fine-tuning: 0.4 WMSE + 0.3 SSIM + 0.3 MAE\n")
            new_lines.append("    return 0.4 * wmse_criterion(pred, target) + 0.3 * ssim_criterion(pred, target) + 0.3 * mae_criterion(pred, target)\n")
            new_lines.append("\n")
            new_lines.append("class NowcastingMetrics:\n")
            new_lines.append("    @staticmethod\n")
            new_lines.append("    def critical_success_index(pred, target, threshold=0.5):\n")
            new_lines.append("        pred_binary = (pred.abs() > threshold).float()\n")
            new_lines.append("        target_binary = (target.abs() > threshold).float()\n")
            new_lines.append("        hits = (pred_binary * target_binary).sum()\n")
            new_lines.append("        misses = ((1 - pred_binary) * target_binary).sum()\n")
            new_lines.append("        false_alarms = (pred_binary * (1 - target_binary)).sum()\n")
            new_lines.append("        return (hits / (hits + misses + false_alarms + 1e-8)).item()\n")
            new_lines.append("\n")
            new_lines.append("    @staticmethod\n")
            new_lines.append("    def probability_of_detection(pred, target, threshold=0.5):\n")
            new_lines.append("        pred_binary = (pred.abs() > threshold).float()\n")
            new_lines.append("        target_binary = (target.abs() > threshold).float()\n")
            new_lines.append("        hits = (pred_binary * target_binary).sum()\n")
            new_lines.append("        misses = ((1 - pred_binary) * target_binary).sum()\n")
            new_lines.append("        return (hits / (hits + misses + 1e-8)).item()\n")
            new_lines.append("\n")
            new_lines.append("metrics = NowcastingMetrics()\n")
            new_lines.append("\n")
            # Proper weight initialization is now in Cell 6
            # new_lines.append("def init_weights(m):\n") ... REMOVED
            new_lines.append("if start_epoch == 0:  # Only init if not resuming\n")
            new_lines.append("    model.apply(init_weights)\n")
            new_lines.append("    print('Applied Kaiming weight initialization')\n")
            new_lines.append("\n")
            new_lines.append("optimizer = optim.Adam(model.parameters(), lr=2e-5)\n")
            new_lines.append("scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)\n")
            new_lines.append("\n")
            new_lines.append("NUM_EPOCHS = 100\n")
            new_lines.append("PATIENCE = 20\n")  # Increased patience for fine-tuning
            new_lines.append("patience_counter = 0\n")
            new_lines.append("best_val_loss = float('inf')\n")
            # new_lines.append("train_losses, val_losses = [], []\n") # Moved to Resume Cell
            new_lines.append("\n")
            new_lines.append("if len(train_losses) > 0 and start_epoch > 0:\n")
            new_lines.append("    print('Restoring training history...')\n")
            new_lines.append("    for i in range(len(train_losses)):\n")
            new_lines.append("        print(f'Epoch {i+1:2d} | Train: {train_losses[i]:.6f} | Val: {val_losses[i]:.6f}')\n")
            new_lines.append("\n")
            new_lines.append("print(f'Training Configuration:')\n")
            new_lines.append("print(f'  Epochs: {NUM_EPOCHS}')\n")
            new_lines.append("print(f'  Batch size: {BATCH_SIZE}')\n")
            new_lines.append("print(f'  Hidden dim: {HIDDEN_DIM}')\n")
            new_lines.append("print(f'  Input: {T_IN}h -> Output: {T_OUT}h')\n")
            new_lines.append("print('=' * 60)\n")
            new_lines.append("\n")
            # --- Loaders ---
            new_lines.append("# Initialize Loaders (Parallel)\n")
            new_lines.append("train_loader = get_loader('train', shuffle=True)\n")
            new_lines.append("val_loader = get_loader('val', shuffle=False)\n")
            new_lines.append("\n")
            # --- Loop ---
            new_lines.append("for epoch in range(start_epoch, NUM_EPOCHS):\n")
            new_lines.append("    epoch_start = time.time()\n")
            new_lines.append("    \n")
            
            # Train
            new_lines.append("    model.train()\n")
            new_lines.append("    train_loss, n_batches = 0.0, 0\n")
            new_lines.append("    pbar = tqdm(train_loader, \n")
            new_lines.append("                total=n_train_batches, desc=f'Epoch {epoch+1} [Train]', leave=False)\n")
            new_lines.append("    \n")
            new_lines.append("    # Debug: Check first layer weights to verify updates\n")
            new_lines.append("    param_before = next(model.parameters()).clone()\n")
            new_lines.append("    \n")
            new_lines.append("    for x, y in pbar:\n")
            new_lines.append("        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)\n")
            new_lines.append("        optimizer.zero_grad(set_to_none=True)\n")
            new_lines.append("        \n")
            new_lines.append("        with torch.amp.autocast('cuda'):\n")
            new_lines.append("            out = model(x, T_OUT)\n")
            new_lines.append("            loss = criterion(out.float(), y.float())\n")
            new_lines.append("        \n")
            new_lines.append("        scaler.scale(loss).backward()\n")
            new_lines.append("        scaler.unscale_(optimizer)\n")
            new_lines.append("        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n")
            new_lines.append("        scaler.step(optimizer)\n")
            new_lines.append("        scaler.update()\n")
            new_lines.append("        \n")
            new_lines.append("        train_loss += loss.item()\n")
            new_lines.append("        n_batches += 1\n")
            new_lines.append("        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'scale': f'{scaler.get_scale():.1e}', 'grad': f'{grad_norm:.2f}'})\n")
            new_lines.append("    \n")
            new_lines.append("    # Verify weight update\n")
            new_lines.append("    param_after = next(model.parameters())\n")
            new_lines.append("    weight_diff = (param_after - param_before).abs().sum().item()\n")
            new_lines.append("    if weight_diff == 0:\n")
            new_lines.append("        print(f'⚠️ WARNING: Weights did not update this epoch! (Scaler scale: {scaler.get_scale()}, Grad norm: {grad_norm})_')\n")
            new_lines.append("    \n")
            new_lines.append("    train_loss /= n_batches if n_batches > 0 else 1\n")
            new_lines.append("    train_losses.append(train_loss)\n")
            new_lines.append("    \n")
            
            # Val
            new_lines.append("    model.eval()\n")
            new_lines.append("    val_loss, val_csi, val_pod, n_val = 0.0, 0.0, 0.0, 0\n")
            new_lines.append("    with torch.no_grad():\n")
            new_lines.append("        pbar = tqdm(val_loader, \n")
            new_lines.append("                             desc=f'Epoch {epoch+1} [Val]', leave=False)\n")
            new_lines.append("        torch.cuda.empty_cache()\n")
            new_lines.append("        for x, y in pbar:\n")
            new_lines.append("            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)\n")
            new_lines.append("            with torch.amp.autocast('cuda'):\n")
            new_lines.append("                pred = model(x, T_OUT)\n")
            new_lines.append("                val_loss += criterion(pred.float(), y.float()).item()\n")
            new_lines.append("                # Metrics (Threshold 0.5 ~ significant rain in z-score)\n")
            new_lines.append("                val_csi += metrics.critical_success_index(pred, y, threshold=0.5)\n")
            new_lines.append("                val_pod += metrics.probability_of_detection(pred, y, threshold=0.5)\n")
            new_lines.append("            n_val += 1\n")
            new_lines.append("    \n")
            new_lines.append("    val_loss /= n_val if n_val > 0 else 1\n")
            new_lines.append("    val_csi /= n_val if n_val > 0 else 1\n")
            new_lines.append("    val_pod /= n_val if n_val > 0 else 1\n")
            new_lines.append("    val_losses.append(val_loss)\n")
            new_lines.append("    scheduler.step()\n")
            new_lines.append("    \n")
            
            # Stats & Save
            new_lines.append("    epoch_time = time.time() - epoch_start\n")
            new_lines.append("    current_lr = optimizer.param_groups[0]['lr']\n")
            new_lines.append("    marker = '\u2605 BEST' if val_loss < best_val_loss else ''\n")
            new_lines.append("    print(f'Epoch {epoch+1:2d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | CSI: {val_csi:.3f} | POD: {val_pod:.3f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s {marker}')\n")
            new_lines.append("    \n")
            # Per-lead-time loss
            new_lines.append("    # Per-lead-time loss breakdown\n")
            new_lines.append("    if (epoch + 1) % 5 == 0 or epoch == 0:\n")
            new_lines.append("        model.eval()\n")
            new_lines.append("        with torch.no_grad():\n")
            new_lines.append("            for x_lt, y_lt in get_loader('val'):\n")
            new_lines.append("                x_lt = x_lt.to(device, non_blocking=True)\n")
            new_lines.append("                y_lt = y_lt.to(device, non_blocking=True)\n")
            new_lines.append("                with torch.amp.autocast('cuda'):\n")
            new_lines.append("                    pred_lt = model(x_lt, T_OUT)\n")
            new_lines.append("                lt_msg = '    Lead-time MSE: '\n")
            new_lines.append("                for t in range(T_OUT):\n")
            new_lines.append("                    lt_mse = nn.MSELoss()(pred_lt[:, t], y_lt[:, t]).item()\n")
            new_lines.append("                    lt_msg += f't+{t+1}h={lt_mse:.4f} '\n")
            new_lines.append("                print(lt_msg)\n")
            new_lines.append("                break  # One batch is enough\n")
            new_lines.append("        model.train()\n")
            new_lines.append("    \n")
            new_lines.append("    if val_loss < best_val_loss:\n")
            new_lines.append("        best_val_loss = val_loss\n")
            new_lines.append("        torch.save({'model': model.state_dict(), 'val_loss': val_loss}, \n")
            new_lines.append("                   f'{CHECKPOINT_DIR}/best_model.pth')\n")
            new_lines.append("        patience_counter = 0\n")
            new_lines.append("    else:\n")
            new_lines.append("        patience_counter += 1\n")
            new_lines.append("    \n")
            new_lines.append("    if (epoch + 1) % 10 == 0:\n")
            new_lines.append("        torch.save({'epoch': epoch, 'model': model.state_dict(),\n")
            new_lines.append("                    'train_losses': train_losses, 'val_losses': val_losses},\n")
            new_lines.append("                   f'{CHECKPOINT_DIR}/checkpoint_e{epoch+1}.pth')\n")
            new_lines.append("    \n")
            new_lines.append("    # Save last model for robust resumption (every epoch)\n")
            new_lines.append("    torch.save({\n")
            new_lines.append("        'epoch': epoch,\n")
            new_lines.append("        'model': model.state_dict(),\n")
            new_lines.append("        'train_losses': train_losses,\n")
            new_lines.append("        'val_losses': val_losses,\n")
            new_lines.append("        'val_loss': val_loss\n")
            new_lines.append("    }, f'{CHECKPOINT_DIR}/last.pth')\n")
            new_lines.append("    print(f'  Saved last.pth (Epoch {epoch+1})')\n")
            new_lines.append("    \n")
            
            # Early Stopping Check
            new_lines.append("    if patience_counter >= PATIENCE:\n")
            new_lines.append("        print(f'\\nEarly stopping triggered at epoch {epoch+1}')\n")
            new_lines.append("        break\n")
            new_lines.append("    \n")
            new_lines.append("    gc.collect()\n")
            new_lines.append("    torch.cuda.empty_cache()\n")
            new_lines.append("\n")
            new_lines.append("print('=' * 60)\n")
            new_lines.append("print(f'\u2713 Training complete! Best val loss: {best_val_loss:.6f}')")
            
            nb['cells'][cell_loop_idx]['source'] = new_lines
            print(f"Updated Training Loop in Cell {cell_loop_idx}")
            break

    # 4. Replace Cell 8: Plot Training Curves (with log-scale + zoomed)
    print("Updating training curve plot cell...")
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        if "Plot Training Curves" in src or ("train_losses" in src and "plt.plot" in src and "Training Progress" in src):
            nb['cells'][i]['source'] = [
                "# Cell 8: Plot Training Curves (Multi-view)\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "\n",
                "# Plot 1: Full scale\n",
                "axes[0].plot(train_losses, label='Train', linewidth=2, color='#2196F3')\n",
                "axes[0].plot(val_losses, label='Val', linewidth=2, color='#FF9800')\n",
                "axes[0].set_xlabel('Epoch')\n",
                "axes[0].set_ylabel('Loss')\n",
                "axes[0].set_title('Full Scale')\n",
                "axes[0].legend()\n",
                "axes[0].grid(alpha=0.3)\n",
                "\n",
                "# Plot 2: Log scale\n",
                "axes[1].plot(train_losses, label='Train', linewidth=2, color='#2196F3')\n",
                "axes[1].plot(val_losses, label='Val', linewidth=2, color='#FF9800')\n",
                "axes[1].set_xlabel('Epoch')\n",
                "axes[1].set_ylabel('Loss (log)')\n",
                "axes[1].set_title('Log Scale')\n",
                "axes[1].set_yscale('log')\n",
                "axes[1].legend()\n",
                "axes[1].grid(alpha=0.3)\n",
                "\n",
                "# Plot 3: Zoomed (skip first 2 epochs)\n",
                "if len(train_losses) > 2:\n",
                "    axes[2].plot(range(2, len(train_losses)), train_losses[2:], label='Train', linewidth=2, color='#2196F3')\n",
                "    axes[2].plot(range(2, len(val_losses)), val_losses[2:], label='Val', linewidth=2, color='#FF9800')\n",
                "axes[2].set_xlabel('Epoch')\n",
                "axes[2].set_ylabel('Loss')\n",
                "axes[2].set_title('Zoomed (Epoch 3+)')\n",
                "axes[2].legend()\n",
                "axes[2].grid(alpha=0.3)\n",
                "\n",
                "plt.suptitle('Training Progress', fontsize=14, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.savefig(f'{FIGURES_DIR}/training_curve.png', dpi=150, bbox_inches='tight')\n",
                "plt.show()\n",
                "print(f'Best val loss: {best_val_loss:.6f}')\n",
                "print(f'Final LR: {optimizer.param_groups[0][\"lr\"]:.2e}')"
            ]
            print(f"  Updated plot cell at index {i}")
            break

    # 5. Replace Cell 9: Sample Prediction (with colorbars + value ranges)
    print("Updating sample prediction cell...")
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        if "Sample Prediction" in src and "imshow" in src:
            nb['cells'][i]['source'] = [
                "# Cell 9: Sample Prediction (with colorbars)\n",
                "model.eval()\n",
                "for x, y in get_loader('val'):\n",
                "    x_sample = x[:1].to(device)\n",
                "    y_sample = y[:1]\n",
                "    break\n",
                "\n",
                "with torch.no_grad():\n",
                "    with torch.amp.autocast('cuda'):\n",
                "        pred = model(x_sample, T_OUT)\n",
                "\n",
                "pred_np = pred[0].cpu().float().numpy()\n",
                "true_np = y_sample[0].numpy()\n",
                "\n",
                "# Print value ranges for diagnosis\n",
                "print(f'Truth  range: [{true_np.min():.4f}, {true_np.max():.4f}], mean={true_np.mean():.4f}')\n",
                "print(f'Pred   range: [{pred_np.min():.4f}, {pred_np.max():.4f}], mean={pred_np.mean():.4f}')\n",
                "\n",
                "# Shared colorscale for fair comparison\n",
                "vmin = min(true_np[:, 0].min(), pred_np[:, 0].min())\n",
                "vmax = max(true_np[:, 0].max(), pred_np[:, 0].max())\n",
                "\n",
                "fig, axes = plt.subplots(2, 3, figsize=(16, 9))\n",
                "lead_times = [0, 2, 5]\n",
                "\n",
                "for col, t in enumerate(lead_times):\n",
                "    # Truth row\n",
                "    im0 = axes[0, col].imshow(true_np[t, 0], cmap='Blues', vmin=vmin, vmax=vmax)\n",
                "    axes[0, col].set_title(f'Truth t+{t+1}h', fontsize=12, fontweight='bold')\n",
                "    plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)\n",
                "    \n",
                "    # Pred row\n",
                "    im1 = axes[1, col].imshow(pred_np[t, 0], cmap='Blues', vmin=vmin, vmax=vmax)\n",
                "    axes[1, col].set_title(f'Pred t+{t+1}h', fontsize=12, fontweight='bold')\n",
                "    plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)\n",
                "    \n",
                "    # Per-lead-time MSE\n",
                "    lt_mse = ((true_np[t] - pred_np[t]) ** 2).mean()\n",
                "    axes[1, col].set_xlabel(f'MSE: {lt_mse:.4f}', fontsize=10)\n",
                "\n",
                "plt.suptitle('Precipitation Forecast (Normalized)', fontsize=14, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.savefig(f'{FIGURES_DIR}/sample_prediction.png', dpi=150, bbox_inches='tight')\n",
                "plt.show()"
            ]
            print(f"  Updated prediction cell at index {i}")
            break

    # 6. Replace Cell 11: Test (improved with per-lead-time breakdown)
    print("Updating test evaluation cell...")
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        if "Test on 2025 Data" in src:
            nb['cells'][i]['source'] = [
                "# Cell 11: Test on 2025 Data (with per-lead-time breakdown)\n",
                "print('Evaluating on test set (includes 2025 data)...')\n",
                "model.eval()\n",
                "test_loss, n_test = 0.0, 0\n",
                "lead_time_losses = [0.0] * T_OUT\n",
                "\n",
                "with torch.no_grad():\n",
                "    for x, y in tqdm(get_loader('test'), total=n_test_batches, desc='Testing'):\n",
                "        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)\n",
                "        with torch.amp.autocast('cuda'):\n",
                "            pred = model(x, T_OUT)\n",
                "            test_loss += criterion(pred, y).item()\n",
                "            for t in range(T_OUT):\n",
                "                lead_time_losses[t] += nn.MSELoss()(pred[:, t], y[:, t]).item()\n",
                "        n_test += 1\n",
                "\n",
                "test_loss /= n_test if n_test > 0 else 1\n",
                "print(f'\\n\u2713 Test Loss (2024-2025 data): {test_loss:.6f}')\n",
                "print(f'\\nPer-lead-time MSE:')\n",
                "for t in range(T_OUT):\n",
                "    lt = lead_time_losses[t] / (n_test if n_test > 0 else 1)\n",
                "    print(f'  t+{t+1}h: {lt:.6f}')\n",
                "\n",
                "# Plot lead-time degradation\n",
                "lt_vals = [lead_time_losses[t] / (n_test if n_test > 0 else 1) for t in range(T_OUT)]\n",
                "plt.figure(figsize=(8, 4))\n",
                "plt.bar(range(1, T_OUT+1), lt_vals, color='#2196F3', alpha=0.8)\n",
                "plt.xlabel('Lead Time (hours)')\n",
                "plt.ylabel('MSE')\n",
                "plt.title('Forecast Skill Degradation by Lead Time')\n",
                "plt.grid(alpha=0.3, axis='y')\n",
                "plt.savefig(f'{FIGURES_DIR}/lead_time_skill.png', dpi=150, bbox_inches='tight')\n",
                "plt.show()"
            ]
            print(f"  Updated test cell at index {i}")
            break

    # 7. Replace Cell 10: Save Final Model (Fix checkpoint key)
    print("Updating final model save cell...")
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        if "Save Final Model" in src and "torch.save" in src:
            nb['cells'][i]['source'] = [
                "# Cell 10: Save Final Model\n",
                "torch.save({\n",
                "    'model': model.state_dict(),  # Key fixed to match evaluate.py\n",
                "    'config': {'hidden_dim': HIDDEN_DIM, 'n_layers': 2, 'T_IN': T_IN, 'T_OUT': T_OUT},\n",
                "    'mean': mean, 'std': std, 'variables': variables,\n",
                "    'best_val_loss': best_val_loss,\n",
                "    'train_losses': train_losses, 'val_losses': val_losses\n",
                "}, f'{CHECKPOINT_DIR}/final_model.pth')\n",
                "\n",
                "print(f'\u2713 Saved to: {CHECKPOINT_DIR}/final_model.pth')\n",
                "print(f'Best validation loss: {best_val_loss:.6f}')"
            ]
            print(f"  Updated final save cell at index {i}")
            break

    with open(OUTPUT_NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print(f"Created {OUTPUT_NB}")

if __name__ == '__main__':
    main()
