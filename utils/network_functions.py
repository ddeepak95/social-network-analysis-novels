def filter_interactions_df(interactions_df, min_cooccurrence=1, min_character_degree=1):
    import pandas as pd
    # Accept JSON list or DataFrame
    if isinstance(interactions_df, list):
        df = pd.DataFrame(interactions_df)
    else:
        df = interactions_df.copy()

    # Filter by edge co-occurrence
    df = df[df['cooccurrences'] >= min_cooccurrence]
    if df.empty:
        return df

    # Count total degree for each character
    char_counts = pd.concat([df['character_1'], df['character_2']]).value_counts()
    eligible_chars = char_counts[char_counts >= min_character_degree].index

    # Keep only rows where both characters meet the degree threshold
    df = df[df['character_1'].isin(eligible_chars) & df['character_2'].isin(eligible_chars)]
    return df

def plot_character_network_with_layout(interactions_df, layout_type='kamada_kawai',
                                       min_cooccurrence=1, top_n_labels=20,
                                       focus_top_n=True, label_size=8, min_character_degree=1):
    

    """
    Plot a character network with a given layout type.

    Args:
        interactions_df (pd.DataFrame or list): DataFrame or list of dictionaries containing character interactions.
        layout_type (str): Layout type: 'kamada_kawai', 'spring', 'shell', 'circular', or 'spectral'.
        min_cooccurrence (int): Minimum number of co-occurrences to include an edge.
        focus_top_n (bool): Whether to focus on the top N characters by degree.
        label_size (int): Font size for character labels.
        min_character_degree (int): Minimum number of co-occurrences for a character to be included in the network.
        top_n_labels (int): Number of top characters to label in the network.
        min_cooccurrence (int): Minimum number of co-occurrences to include an edge.
    
    """


    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import networkx as nx
    import pandas as pd

    # --- Accept JSON list or DataFrame ---
    if isinstance(interactions_df, list):
        # Assume list of dicts (JSON-like)
        df = pd.DataFrame(interactions_df)
    else:
        df = interactions_df

    df = filter_interactions_df(interactions_df, min_cooccurrence, min_character_degree)
    if df.empty:
        print(f"No interactions after filtering (min_cooccurrence={min_cooccurrence}, min_character_degree={min_character_degree}).")
        return
    # Build full graph
    G_full = nx.Graph()
    for _, row in df.iterrows():
        G_full.add_edge(row['character_1'], row['character_2'], weight=row['cooccurrences'])

    # === Identify top N characters by degree ===
    degree_dict = dict(G_full.degree())
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:top_n_labels]

    # === Focus filtering ===
    if focus_top_n:
        nodes_to_keep = set(top_nodes)
        for node in top_nodes:
            nodes_to_keep.update(G_full.neighbors(node))
        G = G_full.subgraph(nodes_to_keep).copy()
    else:
        G = G_full

    if len(G) == 0:
        print("No nodes to visualize after filtering.")
        return

    # === Layout selection ===
    if layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout_type == 'spring':
        pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G)), iterations=500, seed=42)
    elif layout_type == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout_type == 'shell':
        pos = nx.shell_layout(G)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    else:
        raise ValueError(f"Unsupported layout: {layout_type}")

    # Edge weights and log-scaled colors
    weights = nx.get_edge_attributes(G, 'weight')
    edge_weights = np.array([weights[edge] for edge in G.edges()])
    norm = mcolors.LogNorm(vmin=edge_weights.min(), vmax=edge_weights.max())
    grey_green_cmap = mcolors.LinearSegmentedColormap.from_list("GreyGreen", ["#d3d3d3", "#008000"])
    edge_colors = [grey_green_cmap(norm(w)) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw elements
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color="lightgray")
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=[0.4 + np.log1p(w) * 0.3 for w in edge_weights],
        alpha=0.7,
    )

    labels = {node: node for node in G.nodes() if node in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=label_size, ax=ax)

    sm = cm.ScalarMappable(cmap=grey_green_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Co-occurrence Frequency (log scale)", rotation=270, labelpad=15)
    ax.axis('off')
    print("Plotting character network: ", layout_type, "layout")
    plt.tight_layout()
    plt.show()


def plot_ego_network(
    interactions_df, 
    target_character, 
    min_cooccurrence=1, 
    layout_type='kamada_kawai', 
    label_size=7,
    min_character_degree=1
):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    # --- Accept JSON list or DataFrame ---
    if isinstance(interactions_df, list):
        df = pd.DataFrame(interactions_df)
    else:
        df = interactions_df

    df = filter_interactions_df(interactions_df, min_cooccurrence, min_character_degree)
    if df.empty:
        print(f"No interactions after filtering (min_cooccurrence={min_cooccurrence}, min_character_degree={min_character_degree}).")
        return
    
    # Create full graph
    G_full = nx.Graph()
    for _, row in df.iterrows():
        G_full.add_edge(row['character_1'], row['character_2'], weight=row['cooccurrences'])

    if target_character not in G_full:
        print(f"Character '{target_character}' not found in graph.")
        return

    # Create ego graph
    ego_G = nx.ego_graph(G_full, target_character)

    # Choose layout
    if layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(ego_G)
    elif layout_type == 'spring':
        pos = nx.spring_layout(ego_G, seed=42)
    elif layout_type == 'shell':
        pos = nx.shell_layout(ego_G)
    elif layout_type == 'spectral':
        pos = nx.spectral_layout(ego_G)
    elif layout_type == 'circular':
        pos = nx.circular_layout(ego_G)
    else:
        raise ValueError(f"Unsupported layout: {layout_type}")

    # --- Edge weights and log-scaled colors ---
    weights = nx.get_edge_attributes(ego_G, 'weight')
    edge_weights = np.array([weights[edge] for edge in ego_G.edges()])
    if len(edge_weights) == 0:
        print("No edges to plot in ego network.")
        return
    norm = mcolors.LogNorm(vmin=edge_weights.min(), vmax=edge_weights.max())
    grey_green_cmap = mcolors.LinearSegmentedColormap.from_list("GreyGreen", ["#d3d3d3", "#008000"])
    edge_colors = [grey_green_cmap(norm(w)) for w in edge_weights]
    edge_widths = [0.4 + np.log1p(w) * 0.3 for w in edge_weights]

    # Draw ego network
    node_colors = ['skyblue' if n == target_character else 'lightgray' for n in ego_G.nodes()]
    node_sizes = [150 if n == target_character else 10 for n in ego_G.nodes()]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(ego_G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(
        ego_G, pos, ax=ax,
        edge_color=edge_colors,
        width=0.5,
        alpha=0.7,
    )
    nx.draw_networkx_labels(ego_G, pos, font_size=label_size, ax=ax)

    sm = cm.ScalarMappable(cmap=grey_green_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Co-occurrence Frequency (log scale)", rotation=270, labelpad=15)
    ax.set_title(f"Ego Network for '{target_character}'")
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def summarize_character_network(interactions_df, min_cooccurrence=1, top_n=10, tablefmt="simple", min_character_degree=1):
    import networkx as nx
    import pandas as pd
    from tabulate import tabulate

    # Descriptions for global metrics
    metric_descriptions = {
        "Number of Nodes": "Total number of unique characters in the network.",
        "Number of Edges": "Total number of unique character pairs with at least the minimum co-occurrence.",
        "Density": "Proportion of possible connections in the network that are actual connections.",
        "Average Clustering Coefficient": "Average likelihood that a character's connections are also connected to each other.",
        "Transitivity (Global Clustering)": "Overall probability that the adjacent nodes of a node are connected.",
        "Connected Components": "Number of disconnected sub-networks in the graph.",
        "Largest Component Size": "Number of characters in the largest connected sub-network.",
        "Number of Triangles": "Number of groups of three characters all connected to each other.",
        "Average Shortest Path Length": "Average number of steps along the shortest paths for all possible pairs of nodes."
    }

    # Descriptions for centrality measures
    centrality_descriptions = {
        "Degree Centrality": (
            "How many direct connections a character has to others, "
            "relative to the total possible. A higher value means the character interacts with more characters directly."
        ),
        "Betweenness Centrality": (
            "How often a character acts as a bridge along the shortest path between two other characters. "
            "A high value means the character is important for connecting different groups."
        ),
        "Closeness Centrality": (
            "How close a character is to all other characters in the network, based on the shortest paths. "
            "A higher value means the character can quickly interact with everyone else."
        ),
        "Eigenvector Centrality": (
            "A measure of a character's influence, considering not just their direct connections, "
            "but also how well-connected their friends are. High values mean the character is connected to other important characters."
        ),
        "Triangles": (
            "The number of groups of three characters (including this one) where everyone is connected to each other. "
            "A higher number means the character is part of many tightly-knit groups."
        )
    }
    # --- Accept JSON list or DataFrame ---
    if isinstance(interactions_df, list):
        df = pd.DataFrame(interactions_df)
    else:
        df = interactions_df

    df = filter_interactions_df(interactions_df, min_cooccurrence, min_character_degree)
    if df.empty:
        print(f"No interactions after filtering (min_cooccurrence={min_cooccurrence}, min_character_degree={min_character_degree}).")
        return
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['character_1'], row['character_2'], weight=row['cooccurrences'])

    if G.number_of_nodes() == 0:
        print("Graph is empty after filtering.")
        return None

    # --- Global Network Metrics ---
    global_metrics = {
        "Number of Nodes": G.number_of_nodes(),
        "Number of Edges": G.number_of_edges(),
        "Density": nx.density(G),
        "Average Clustering Coefficient": nx.average_clustering(G),
        "Transitivity (Global Clustering)": nx.transitivity(G),
        "Connected Components": nx.number_connected_components(G),
        "Largest Component Size": len(max(nx.connected_components(G), key=len)),
        "Number of Triangles": sum(nx.triangles(G).values()) // 3
    }
    if nx.is_connected(G):
        global_metrics["Average Shortest Path Length"] = nx.average_shortest_path_length(G)
    else:
        global_metrics["Average Shortest Path Length"] = "Graph not connected"

    # Print global metrics as a table with descriptions
    print("=== Global Network Metrics ===")
    global_table = [
        [metric, value, metric_descriptions.get(metric, "")]
        for metric, value in global_metrics.items()
    ]
    print(tabulate(global_table, headers=["Metric", "Value", "Description"], tablefmt=tablefmt))

    # --- Centrality Measures ---
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector = {n: 0 for n in G.nodes()}

    centrality_df = pd.DataFrame({
        "Character": list(G.nodes()),
        "Degree Centrality": [degree_centrality[n] for n in G.nodes()],
        "Betweenness Centrality": [betweenness[n] for n in G.nodes()],
        "Closeness Centrality": [closeness[n] for n in G.nodes()],
        "Eigenvector Centrality": [eigenvector[n] for n in G.nodes()],
        "Triangles": [nx.triangles(G, n) for n in G.nodes()]
    })

    # Print top characters by each centrality as tables with descriptions
    for col in ["Degree Centrality", "Betweenness Centrality", "Eigenvector Centrality", "Triangles"]:
        print(f"\n=== Top Characters by {col} ===")
        print(f"Description: {centrality_descriptions[col]}")
        # Reorder columns: centrality in focus first, then others
        cols = [col] + [c for c in centrality_df.columns if c not in [col, "Character"]]
        cols = ["Character"] + cols  # Always keep Character as the first column
        print(tabulate(
            centrality_df.sort_values(col, ascending=False).head(top_n)[cols],
            headers="keys", showindex=False, tablefmt=tablefmt, floatfmt=".4f"
        ))



