import matplotlib.pyplot as plt
import numpy as np

# Visual Style Guide Enforcement
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3
})
import matplotlib.dates as mdates
from datetime import datetime

def generate_timeline():
    """
    Generates a Gantt-style timeline of key research milestones.
    """
    # Data: (Date, description, category)
    events = [
        ("2015-06-01", "ConvLSTM (Shi et al.)\n[Spatio-Temporal]", "Method"),
        ("2017-06-01", "TrajGRU (Shi et al.)\n[Motion Invariance]", "Method"),
        ("2019-12-01", "U-Net Baseline (Agrawal)\n[CNN-only]", "Baseline"),
        ("2020-03-01", "MetNet (Google)\n[Attention]", "Method"),
        ("2021-01-01", "ERA5-Land Released\n[High-Res Data]", "Data"),
        ("2021-07-01", "DGMR (DeepMind)\n[Generative AI]", "Method"),
        ("2023-01-01", "GraphCast/Pangu\n[Global Foundation]", "Method"),
        ("2024-02-01", "Paper Weather (Ours)\n[Regional Expert]", "Ours"),
    ]

    dates = [datetime.strptime(d[0], "%Y-%m-%d") for d in events]
    names = [d[1] for d in events]
    categories = [d[2] for d in events]

    # Map categories to colors (from Visual Style Guide)
    color_map = {
        "Method": "#E69F00",   # Baseline-like
        "Baseline": "#999999", # Neutral
        "Data": "#0072B2",     # Ground Truth-like
        "Ours": "#009E73"      # Ours
    }
    colors = [color_map[c] for c in categories]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create stem plot
    ax.vlines(dates, 0, [5, 4, 3, 4, 6, 5, 5, 7], color=colors, linewidth=2, alpha=0.7)
    ax.scatter(dates, [5, 4, 3, 4, 6, 5, 5, 7], s=100, color=colors, zorder=3)
    
    # Annotate
    for i, txt in enumerate(names):
        ax.annotate(txt, (dates[i], [5, 4, 3, 4, 6, 5, 5, 7][i]), 
                    xytext=(0, 10), textcoords="offset points", 
                    ha='center', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], alpha=0.9))

    # Format X Axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Style
    ax.set_ylim(0, 9)
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.title("Evolution of Precipitation Nowcasting (2015-2024)", fontsize=14)
    plt.tight_layout()
    
    output_path = "Paper/figures/timeline_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"Timeline saved to {output_path}")

if __name__ == "__main__":
    generate_timeline()
