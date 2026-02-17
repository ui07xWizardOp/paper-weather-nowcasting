# Visual Hierarchy Audit (Phase 30)

## 1. Global Aesthetics
-   [x] **Color Palette**: Consistent usage of `#009E73` (Green) for "Ours" and `#E69F00` (Orange) for "Baseline" across all 3 plotting scripts.
-   [x] **Typography**: All Figures use `Arial` (Sans-Serif) to contrast with the `Times` (Serif) body text.
-   [x] **Readability**: Minimum font size set to 8pt via `rcParams`.

## 2. Manuscript Layout (`main.tex`)
-   [x] **Figure Placement**: All floats use `[t!]` (Top) to respect the "Reading Gravity" and prevent text fragmentation.
-   [x] **Table Header**: `\caption` is strictly *above* the table (IEEE Standard).
-   [x] **Table Rules**: `\toprule`, `\midrule`, `\bottomrule` used (No vertical bars).

## 3. Element Presence Check
-   [x] **Abstract**: Present and < 250 words.
-   [x] **Keywords**: Present and optimized.
-   [x] **Introduction**: Starts with a "Hook" (Indian Monsoon).
-   [x] **Methodology**: Contains Algorithm 1 (Pseudocode).
-   [x] **Results**: Contains Table 2 (metrics) and Figure 3 (Qualitative).
-   [x] **Discussion**: Distinct from Conclusion.
-   [x] **References**: 30+ Citations formatted in BibTeX.

## 4. Final Polish
-   [x] **Capitalization**: Section headings use Title Case.
-   [x] **Hyperlinks**: `hyperref` package enabled for clickable refs.
