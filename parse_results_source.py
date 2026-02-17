import json
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

def parse_notebook_source(filepath):
    try:
        print(f"Opening {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = "".join(cell['source'])
                if 'WeatherNowcaster' in source or 'HIDDEN_DIM' in source:
                    print(f"\n--- Cell {i} ---")
                    print(source)
                    print("-" * 20)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parse_notebook_source(r"c:\Users\KIIT0001\Desktop\projects\Paper Weather\results\v2\03_training_optimized.ipynb")
