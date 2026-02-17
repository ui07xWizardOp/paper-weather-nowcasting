# Figure Quality Checklist

## General
- [ ] **DPI**: All raster images (heatmaps) are >300 DPI.
- [ ] **Fonts**: Axis labels are Arial/Helvetica, size > 12pt (readable when scaled down).
- [ ] **Colorblind Safe**: Used distinct colormaps (e.g., Viridis/Magma) instead of Jet/Rainbow.

## Figure 1: Architecture
- [ ] Clearly labels "Encoder" (Input) vs "Decoder" (Prediction).
- [ ] Shows "Copy States" (h, c) flowing horizontally.

## Figure 2: Study Area
- [ ] Includes scale bar (km) and North arrow.
- [ ] Lat/Lon ticks are visible but unintrusive.

## Figure 3: Qualitative Results (The "Strip")
- [ ] Time steps ($t+1, t+3, t+6$) labeled on top.
- [ ] Rows labeled: Ground Truth, Ours, Baseline.
- [ ] "Difference Map" included to highlight error distribution?
