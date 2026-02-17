# Visual Style Guide: Paper Weather

## Philosophy
"Glance Value" is paramount. A reviewer should understand the paper's contribution just by scanning the figures.
-   **Cleanliness**: Minimal chartjunk (no gridlines unless necessary).
-   **Consistency**: Same colors for same concepts across all figures.
-   **Legibility**: All text must be readable when printed on A4 paper (min 8pt).

## 1. Color Palette (Colorblind Safe)
We use a modification of the **Okabe-Ito** palette.

| Concept | Color Hex | Matplotlib Name | Usage |
| :--- | :--- | :--- | :--- |
| **Ours (Weighted MSE)** | `#009E73` | `green` (approx) | The proposed method (Hero Color). |
| **Baseline (MSE)** | `#E69F00` | `orange` | The standard ConvLSTM baseline. |
| **Ground Truth** | `#0072B2` | `blue` | ERA5-Land observed data. |
| **Positive Difference** | `#56B4E9` | `skyblue` | Improvements. |
| **Negative Difference** | `#D55E00` | `vermillion` | Errors/Artifacts. |

## 2. Typography
-   **Context**: IEEE Conference/Journal (Two Column).
-   **Font Family**: sans-serif (`Arial` or `Helvetica`) for figures to contrast with Serif body text.
-   **Sizes**:
    -   Title: 12pt (Bold)
    -   Axis Labels: 10pt
    -   Tick Labels: 8pt
    -   Legend: 8pt

## 3. Matplotlib `rcParams` Configuration
```python
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
    "grid.alpha": 0.3,
    "lines.linewidth": 2
})
```

## 4. Figure Layout Rules
-   **Panels**: Labeled (a), (b), (c) in **bold**, top-left corner.
-   **Aspect Ratio**: Golden Ratio ($1.618$) where possible.
-   **Captions**: Must be self-contained (describe *what* is shown, not just *data source*).
