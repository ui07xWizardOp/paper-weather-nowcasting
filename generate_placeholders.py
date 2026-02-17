import os
import base64

# Simple 1x1 pixel white PNG
PLACEHOLDER_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="

FIGURES = [
    "Paper/figures/timeline_plot.png",
    "Paper/figures/citation_graph.png",
    "Paper/figures/figure3_qualitative.png",
    "Paper/figures/figure4_metrics.png"
]

def create_placeholders():
    os.makedirs("Paper/figures", exist_ok=True)
    
    data = base64.b64decode(PLACEHOLDER_B64)
    
    for fig in FIGURES:
        if not os.path.exists(fig):
            print(f"Creating placeholder for {fig}")
            with open(fig, "wb") as f:
                f.write(data)
        else:
            print(f"File {fig} already exists. Skipping.")

if __name__ == "__main__":
    create_placeholders()
