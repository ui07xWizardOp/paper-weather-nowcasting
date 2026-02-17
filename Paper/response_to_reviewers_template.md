# Response to Reviewers Strategy

## General Tone
-   **Polite & Grateful**: "We thank Reviewer #X for their insightful comments..."
-   **Firm on Scope**: "While interesting, X is outside the scope of this regional study."
-   **Evidence-Based**: "We have added Table Y / Figure Z to address this."

## Anticipated Critique 1: "ConvLSTM is old (2015). Why not Diffusion Models?"
-   **Response**: "While Diffusion Models (and Transformers) represent the SOTA, they require massive distinct compute resources not available in many operational centers in the Global South. We deliberately chose ConvLSTM for its balance of efficiency and performance on limited hardware (T4 GPU), proving that architectural tuning (Weighted MSE) matters more than raw model scale for this specific task."

## Anticipated Critique 2: "Is ERA5-Land 'Ground Truth'? It's a model."
-   **Response**: "We explicitly acknowledge this in Section \ref{sec:background} ('ERA5-Land Physics'). However, in data-sparse regions like the Himalayas, ERA5-Land acts as the best available proxy, offering physical consistency that interpolated gauge data cannot match."

## Anticipated Critique 3: "Did you compare with GraphCast?"
-   **Response**: "GraphCast operates at 0.25$^\circ$ resolution. Our target is 0.1$^\circ$. Direct comparison requires downscaling, which introduces artifacts. We compare against baselines operative at the *same* native resolution (Optical Flow)."
