
import json
import os

NOTEBOOK_PATH = 'notebooks/03_training.ipynb'
OUTPUT_PATH = 'notebooks/03_training_resume.ipynb'

# Define the code to be injected as a list of strings
resume_code = [
    "# Cell: Auto-Resume Logic & Drive Mounting\n",
    "import os\n",
    "import torch\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    print('✓ Google Drive mounted')\n",
    "    CHECKPOINT_DIR = '/content/drive/MyDrive/WeatherPaper/checkpoints'\n",
    "    os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
    "    print(f'✓ Checkpoints will be saved to: {CHECKPOINT_DIR}')\n",
    "except ImportError:\n",
    "    print('⚠️ Not running in Colab, using local paths')\n",
    "    CHECKPOINT_DIR = 'checkpoints'\n",
    "\n",
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
    "    checkpoint = torch.load(RESUME_FROM, map_location='cuda' if torch.cuda.is_available() else 'cpu')\n",
    "    if 'model' in checkpoint:\n",
    "        model.load_state_dict(checkpoint['model'])\n",
    "    else:\n",
    "        model.load_state_dict(checkpoint)\n",
    "    \n",
    "    if 'epoch' in checkpoint:\n",
    "        start_epoch = checkpoint['epoch'] + 1\n",
    "        print(f'Resuming from epoch {start_epoch}')\n",
    "    else:\n",
    "        print('Warning: Epoch info not found, fine-tuning from 0')\n"
]

def main():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Locate Training Loop Cell (Cell 7 - Index 7 in 0-indexed list based on previous view)
    target_idx = -1
    for i, cell in enumerate(nb['cells']):
        if "Cell 7: Training Loop" in "".join(cell.get('source', [])):
            target_idx = i
            break
            
    if target_idx != -1:
        # Insert resume logic BEFORE the loop cell
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": resume_code
        }
        # Check if we already inserted it to avoid duplicates if re-run
        prev_cell = nb['cells'][target_idx-1] if target_idx > 0 else {}
        if "Auto-Resume Logic" in "".join(prev_cell.get('source', [])):
            print("Resume logic already present, updating it.")
            nb['cells'][target_idx-1] = new_cell
        else:
            nb['cells'].insert(target_idx, new_cell)
            target_idx += 1 # Update index since we inserted
        
        # Modify the loop cell to use start_epoch
        loop_cell = nb['cells'][target_idx]
        source = loop_cell['source']
        for i, line in enumerate(source):
            if "for epoch in range(NUM_EPOCHS):" in line:
                source[i] = "for epoch in range(start_epoch, NUM_EPOCHS):\n"
                break
        
        print(f"Successfully injected resume logic.")

        # Inject 'Save Last' logic at end of loop
        # Loop ends at 'torch.cuda.empty_cache()' inside the loop cell
        save_last_code = [
            "    # Save last model for robust resumption (every epoch)\n",
            "    torch.save({\n",
            "        'epoch': epoch,\n",
            "        'model': model.state_dict(),\n",
            "        'train_losses': train_losses,\n",
            "        'val_losses': val_losses,\n",
            "        'val_loss': val_loss\n",
            "    }, f'{CHECKPOINT_DIR}/last.pth')\n",
            "    print(f'  Saved last.pth (Epoch {epoch+1})')\n",
            "    \n"
        ]
        
        # Find where to insert (before gc.collect() at end of loop)
        insert_idx = -1
        for i, line in enumerate(source):
            if "gc.collect()" in line and "    " in line: # Indented gc.collect
                insert_idx = i
                break
        
        if insert_idx != -1:
            # Check if already inserted
            if "last.pth" not in "".join(source):
                for line in reversed(save_last_code):
                    source.insert(insert_idx, line)
                print("Successfully injected 'save last.pth' logic.")
        else:
            print("Warning: Could not find end of loop to insert save logic.")
    else:
        print("Could not find 'Cell 7: Training Loop', appending to end.")
        nb['cells'].append({
            "cell_type": "code", 
            "execution_count": None,
            "metadata": {}, 
            "outputs": [],
            "source": resume_code
        })

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print(f"Created {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
