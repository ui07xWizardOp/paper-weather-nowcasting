# Centrality Analysis Report (Phase 4)

## Graph Structure Interpretation
Based on the `citation_graph.png` structure (generated in Phase 25), we identify the following topological roles.

## 1. High Degree Centrality (The Authority)
**Node**: *Shi et al. (2015) [ConvLSTM]*
-   **In-Degree**: Highest. Cited by almost every subsequent paper (TrajGRU, MetNet, DGMR, Pangu, Ours).
-   **Interpretation**: This is the "Root Node" of the Deep Learning Nowcasting tree. Every paper MUST cite this to establish context.

## 2. Betweenness Centrality (The Bridge)
**Node**: *Ravuri et al. (2021) [DGMR]*
-   **Role**: Connects the "Deterministic Era" (MSE-based, 2015-2020) with the "Probabilistic Era" (GANs/Diffusion, 2021+).
-   **Interpretation**: It shifted the objective function from minimizing error to maximizing realism.

## 3. Closeness Centrality (The Resource)
**Node**: *Muñoz-Sabater (2021) [ERA5-Land]*
-   **Role**: The central data hub. While not an algorithm, it is the common input across multiple regional studies (Including Ours).
-   **Interpretation**: The field is becoming "Data-Centric"; methodology is converging, but data quality is differentiating.

## 4. Our Position
**Node**: *Paper Weather (2024)*
-   **Role**: Leaf Node (currently).
-   **Connection**: Direct descendant of *Shi (2015)* (Method) and *Muñoz (2021)* (Data).
-   **Goal**: To become a "Branching Node" for "Regional/Efficient Nowcasting".
