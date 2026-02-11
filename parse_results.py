import json
import sys

def parse_notebook(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        print(f"Parsing {filepath}...")
        for i, cell in enumerate(nb['cells']):
            if cell.get('cell_type') == 'code':
                outputs = cell.get('outputs', [])
                if outputs:
                    print(f"\n--- Cell {i} Outputs ---")
                    for output in outputs:
                        # Standard stdout/stderr
                        if output.get('name') in ['stdout', 'stderr']:
                            print(f"[{output['name']}]")
                            print("".join(output.get('text', [])))
                        
                        # Execute result (e.g. return values)
                        elif output.get('output_type') == 'execute_result':
                            data = output.get('data', {})
                            if 'text/plain' in data:
                                print("[result]")
                                print("".join(data['text/plain']))
                        
                        # Display data (e.g. pandas dataframes, plots)
                        elif output.get('output_type') == 'display_data':
                            data = output.get('data', {})
                            if 'text/plain' in data:
                                print("[display]")
                                print("".join(data['text/plain']))
                            # We can't print images but we can note they exist
                            if 'image/png' in data:
                                print("[image/png present]")
                                
    except Exception as e:
        print(f"Error parsing notebook: {e}")

if __name__ == "__main__":
    parse_notebook(r"c:\Users\KIIT0001\Desktop\projects\Paper Weather\results\v2\03_training_optimized.ipynb")
