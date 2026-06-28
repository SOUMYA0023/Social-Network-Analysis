"""
export.py — Export Networks to Gephi-Compatible Formats
========================================================
Exports all graphs as .gexf and .graphml files with node and edge attributes
suitable for visualization in Gephi.

Node attributes included: label/category, degree, centrality scores, community ID
Edge attributes included: weight

Usage:
    python -m src.export
"""

import os
import networkx as nx

from src.build_graph import load_graphs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRAPH_DIR = os.path.join("outputs", "graphs")


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def sanitize_node_attributes(G: nx.Graph) -> nx.Graph:
    """
    Ensure all node attributes are serializable for GEXF/GraphML export.
    Convert lists, sets, and other non-primitive types to strings.
    """
    for node in G.nodes():
        for key, value in G.nodes[node].items():
            if isinstance(value, (list, set, tuple)):
                G.nodes[node][key] = str(value)
            elif isinstance(value, float):
                G.nodes[node][key] = round(value, 6)
    return G


def add_degree_attribute(G: nx.Graph) -> nx.Graph:
    """Add degree as a node attribute for Gephi sizing."""
    for node in G.nodes():
        G.nodes[node]["degree"] = G.degree(node)
    return G


def export_graph(G: nx.Graph, name: str, output_dir: str) -> None:
    """
    Export a graph to both GEXF and GraphML formats.

    Parameters
    ----------
    G : networkx.Graph
    name : base filename (without extension)
    output_dir : directory to save files
    """
    ensure_dir(output_dir)

    # Prepare graph for export
    G_export = G.copy()
    G_export = add_degree_attribute(G_export)
    G_export = sanitize_node_attributes(G_export)

    # Export GEXF (preferred Gephi format)
    gexf_path = os.path.join(output_dir, f"{name}.gexf")
    nx.write_gexf(G_export, gexf_path)
    print(f"  Exported: {gexf_path}  "
          f"({G_export.number_of_nodes()} nodes, {G_export.number_of_edges()} edges)")

    # Export GraphML (alternative format, also Gephi-compatible)
    graphml_path = os.path.join(output_dir, f"{name}.graphml")
    nx.write_graphml(G_export, graphml_path)
    print(f"  Exported: {graphml_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Export all graphs to Gephi formats."""
    print("[export] Loading graphs ...")
    graphs = load_graphs()

    print("\n[export] Exporting all graphs to GEXF and GraphML ...\n")

    # Export each graph with a descriptive filename
    export_names = {
        "keyword_depressive":              "keyword_depressive",
        "keyword_nondepressive":           "keyword_nondepressive",
        "tweet_similarity_depressive":     "tweet_similarity_depressive",
        "tweet_similarity_nondepressive":  "tweet_similarity_nondepressive",
        "combined_keyword":                "combined_keyword",
    }

    for graph_key, filename in export_names.items():
        G = graphs[graph_key]
        print(f"\n  --- {filename} ---")
        export_graph(G, filename, GRAPH_DIR)

    print(f"\n[export] All graphs exported to {GRAPH_DIR}/")
    print("[export] You can now open these files in Gephi.")
    print("[export] Run `python -m src.visualize` next for plots.")


if __name__ == "__main__":
    main()
