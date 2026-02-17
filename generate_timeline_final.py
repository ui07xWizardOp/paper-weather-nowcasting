"""
Generate a publication-quality timeline figure for precipitation nowcasting evolution.
Uses only PIL - no matplotlib dependency.
"""
from PIL import Image, ImageDraw, ImageFont
import os

# --- Configuration ---
OUTPUT = os.path.join("Paper", "figures", "timeline_plot.png")
W, H = 2400, 900  # High-res for print quality
DPI_SCALE = 3      # Effective ~300 DPI at column width

# --- Color Palette (Colorblind-friendly, print-safe) ---
BG        = "#FFFFFF"
AXIS_CLR  = "#2C3E50"
GRID_CLR  = "#ECF0F1"
TEXT_CLR   = "#2C3E50"
LABEL_CLR  = "#7F8C8D"

# Category colors
CAT_COLORS = {
    "NWP":      "#3498DB",   # Blue
    "RNN":      "#E67E22",   # Orange
    "CNN":      "#2ECC71",   # Green
    "GAN":      "#9B59B6",   # Purple
    "Foundation": "#1ABC9C", # Teal
    "Ours":     "#E74C3C",   # Red (highlight)
}

# --- Timeline Data ---
# (start_year, end_year, label, category, is_ours)
EVENTS = [
    (2010, 2025, "NWP (ERA5)", "NWP", False),
    (2015, 2016.5, "ConvLSTM", "RNN", False),
    (2017, 2018.5, "TrajGRU", "RNN", False),
    (2017, 2018.5, "PredRNN", "RNN", False),
    (2019, 2020.5, "U-Net", "CNN", False),
    (2020, 2021.5, "MetNet", "CNN", False),
    (2021, 2022.5, "DGMR", "GAN", False),
    (2023, 2024, "GraphCast", "Foundation", False),
    (2023, 2024, "Pangu", "Foundation", False),
    (2024, 2025.5, "Ours (WMSE-ConvLSTM)", "Ours", True),
]

# Assign rows to avoid overlap
ROW_MAP = {
    "NWP (ERA5)": 0,
    "ConvLSTM": 1,
    "TrajGRU": 2,
    "PredRNN": 3,
    "U-Net": 1,
    "MetNet": 2,
    "DGMR": 1,
    "GraphCast": 2,
    "Pangu": 3,
    "Ours (WMSE-ConvLSTM)": 1,
}

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def draw_rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    x1, y1, x2, y2 = xy
    # Draw rounded rectangle using arcs and rectangles
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
    draw.pieslice([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=fill)
    draw.pieslice([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=fill)
    draw.pieslice([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=fill)
    draw.pieslice([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=fill)
    if outline:
        draw.arc([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=outline, width=width)
        draw.arc([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=outline, width=width)
        draw.arc([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=outline, width=width)
        draw.arc([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=outline, width=width)
        draw.line([x1 + radius, y1, x2 - radius, y1], fill=outline, width=width)
        draw.line([x1 + radius, y2, x2 - radius, y2], fill=outline, width=width)
        draw.line([x1, y1 + radius, x1, y2 - radius], fill=outline, width=width)
        draw.line([x2, y1 + radius, x2, y2 - radius], fill=outline, width=width)


def generate():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # --- Layout constants ---
    margin_left = 120
    margin_right = 80
    margin_top = 120
    margin_bottom = 120
    chart_w = W - margin_left - margin_right
    chart_h = H - margin_top - margin_bottom

    start_year = 2010
    end_year = 2026
    years = end_year - start_year
    px_per_year = chart_w / years

    y_axis_top = margin_top
    y_axis_bottom = margin_top + chart_h
    x_axis_left = margin_left
    x_axis_right = margin_left + chart_w

    bar_h = 50
    row_spacing = 70
    bar_area_top = y_axis_top + 60

    # --- Title ---
    title = "Evolution of Precipitation Nowcasting Methods"
    # Approximate centering
    title_w = len(title) * 14
    draw.text(((W - title_w) // 2, 30), title, fill=TEXT_CLR)

    # --- Draw grid lines & year labels ---
    for i in range(years + 1):
        year = start_year + i
        x = x_axis_left + int(i * px_per_year)
        # Vertical grid
        draw.line([(x, y_axis_top), (x, y_axis_bottom)], fill=GRID_CLR, width=1)
        # Tick
        draw.line([(x, y_axis_bottom), (x, y_axis_bottom + 10)], fill=AXIS_CLR, width=2)
        # Label
        label = str(year)
        lw = len(label) * 7
        draw.text((x - lw // 2, y_axis_bottom + 18), label, fill=LABEL_CLR)

    # --- Draw main axis ---
    draw.line([(x_axis_left, y_axis_bottom), (x_axis_right, y_axis_bottom)], fill=AXIS_CLR, width=3)

    # --- Draw events ---
    for (s, e, label, cat, is_ours) in EVENTS:
        row = ROW_MAP[label]
        color = CAT_COLORS[cat]
        rgb = hex_to_rgb(color)

        x1 = x_axis_left + int((s - start_year) * px_per_year)
        x2 = x_axis_left + int((e - start_year) * px_per_year)
        y1 = bar_area_top + row * row_spacing
        y2 = y1 + bar_h

        # Draw rounded bar
        radius = 8
        draw_rounded_rect(draw, (x1, y1, x2, y2), radius, fill=color)

        # Outline for "Ours"
        if is_ours:
            draw_rounded_rect(draw, (x1, y1, x2, y2), radius, fill=color, outline="#C0392B", width=4)
            # Add a subtle glow effect by drawing a slightly larger rect behind
            # (already done with the outline)

        # Text label inside bar
        text_x = x1 + 12
        text_y = y1 + (bar_h - 14) // 2
        # Check if text fits
        text_w = len(label) * 8
        if text_w < (x2 - x1 - 20):
            draw.text((text_x, text_y), label, fill="white")
        else:
            # Draw above the bar
            draw.text((x1, y1 - 20), label, fill=color)

    # --- Draw legend ---
    legend_x = x_axis_left + 20
    legend_y = y_axis_bottom - 180
    legend_items = [
        ("NWP / Reanalysis", "NWP"),
        ("RNN-based", "RNN"),
        ("CNN-based", "CNN"),
        ("GAN-based", "GAN"),
        ("Foundation Models", "Foundation"),
        ("This Work", "Ours"),
    ]

    # Legend background
    lbg_w = 260
    lbg_h = len(legend_items) * 30 + 20
    draw.rectangle(
        [legend_x - 10, legend_y - 10, legend_x + lbg_w, legend_y + lbg_h],
        fill="#F8F9FA", outline="#BDC3C7", width=1
    )

    for i, (lbl, cat) in enumerate(legend_items):
        y = legend_y + i * 30
        color = CAT_COLORS[cat]
        draw.rectangle([legend_x, y, legend_x + 20, y + 16], fill=color)
        draw.text((legend_x + 30, y), lbl, fill=TEXT_CLR)

    # --- Arrow annotation pointing to "Ours" ---
    ours_x = x_axis_left + int((2024.7 - start_year) * px_per_year)
    ours_y = bar_area_top + 1 * row_spacing - 10
    # Draw annotation text above
    ann_text = "Proposed Method"
    draw.text((ours_x - 40, ours_y - 40), ann_text, fill="#E74C3C")
    # Small arrow line
    draw.line([(ours_x, ours_y - 20), (ours_x, ours_y)], fill="#E74C3C", width=2)
    # Arrowhead
    draw.polygon([(ours_x - 5, ours_y - 5), (ours_x + 5, ours_y - 5), (ours_x, ours_y)], fill="#E74C3C")

    # --- Save ---
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    img.save(OUTPUT, "PNG", dpi=(300, 300))
    print(f"Generated {OUTPUT} ({W}x{H} @ 300 DPI)")

if __name__ == "__main__":
    generate()
