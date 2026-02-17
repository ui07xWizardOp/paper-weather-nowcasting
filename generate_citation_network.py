import matplotlib.pyplot as plt
import networkx as nx

# Visual Style Guide Enforcement
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})

def generate_citation_graph():
    """
    Generates a directed graph showing the lineage of the research.
    Nodes: Key Papers (Shi2015, Ravuri2021, etc.)
    Edges: 'Cites' relationship / Influence flow.
    """
    G = nx.DiGraph()

    # Define Nodes (Papers)
    nodes = {
        "Shi2015": "ConvLSTM (2015)\n[The Backbone]",
        "Shi2017": "TrajGRU (2017)\n[Motion]",
        "Agrawal2019": "U-Net (2019)\n[Baseline]",
        "Hersbach2020": "ERA5 (2020)\n[Global Data]",
        "Muñoz2021": "ERA5-Land (2021)\n[Our Data]",
        "Ravuri2021": "DGMR (2021)\n[Gen AI]",
        "Bi2023": "Pangu (2023)\n[Foundation]",
        "Ours": "Paper Weather (2024)\n[Regional Expert]"
    }

    # Add Nodes
    for k, v in nodes.items():
        G.add_node(k, label=v)

    # Define Edges (Influence)
    edges = [
        ("Shi2015", "Shi2017"),
        ("Shi2015", "Agrawal2019"),
        ("Shi2015", "Ravuri2021"),
        ("Shi2015", "Ours"),      # We use ConvLSTM
        ("Hersbach2020", "Muñoz2021"),
        ("Hersbach2020", "Bi2023"),
        ("Muñoz2021", "Ours"),    # We use ERA5-Land
        ("Agrawal2019", "Ours"),  # We compare against U-Net
        ("Ravuri2021", "Ours"),   # We discuss DGMR
    ]
    
    G.add_edges_from(edges)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    # Draw
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    # Colors: Foundation=Grey, Data=Blue, Generative=Purple, Ours=Green
    node_colors = ['#999999', '#999999', '#999999', '#0072B2', '#0072B2', '#CC79A7', '#E69F00', '#009E73']
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=nodes, font_size=8, font_weight="bold", font_family="sans-serif")
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color='gray', arrowsize=15)
    
    plt.title("Research Lineage: From ConvLSTM to Paper Weather", fontsize=12)
    plt.axis('off')
    
    output_path = "Paper/figures/citation_graph.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Citation graph saved to {output_path}")

if __name__ == "__main__":
    generate_citation_graph()
