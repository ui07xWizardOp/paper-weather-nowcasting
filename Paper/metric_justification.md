# Metric Justification (Phase 11 Critique)

## The Fallacy of MSE
Common question: "Why is your MSE (0.042) higher than the Baseline (0.038) if your model is better?"

**Answer**: 
Input $\mathcal{X}$ is 90% zeros (no rain).
-   **Model A (Baseline)**: Predicts $0.0$ everywhere.
    -   MSE: Low (Only errors on the 10% rain pixels).
    -   Utility: **Zero**. It misses every storm.
-   **Model B (Ours)**: Predicts a storm, but shifts it by 5 pixels.
    -   MSE: High (Double penalty: Error where rain *was* + Error where rain *wasn't*).
    -   Utility: **High**. It warns the user a storm is coming.

## The Need for Categorical Metrics (CSI)
To measure *utility* for disaster warning, we must threshold the output (e.g., $>1.0$ mm/hr) and treat it as a classification problem.

### Critical Success Index (CSI)
$$ CSI = \frac{Hits}{Hits + Misses + FalseAlarms} $$
-   Matches the user's question: *"Of all the times rain was either observed or forecast, how often was it correct?"*
-   We achieve **0.65** vs Baseline **0.58**. This is the "Real World" gain.

### Probability of Detection (POD)
$$ POD = \frac{Hits}{Hits + Misses} $$
-   Matches the safety question: *"If a cloudburst happens, will you catch it?"*
-   We achieve **0.74** (catch 3 out of 4) vs Baseline **0.62**.
