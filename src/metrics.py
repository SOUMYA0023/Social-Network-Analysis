"""
metrics.py — SNA Metric Computations
======================================
Computes comprehensive Social Network Analysis metrics for all graphs:
- Basic network statistics
- Centrality measures (degree, betweenness, closeness, PageRank)
- Clustering and community detection (Louvain)
- Cross-network comparison
- Bridge nodes and exclusive keywords

Usage:
    python -m src.metrics
"""

import os
import pandas as pd
import networkx as nx
import community as community_louvain   # python-louvain library

from src.build_graph import load_graphs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METRICS_DIR = os.path.join("outputs", "metrics")


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Basic Network Statistics
# ---------------------------------------------------------------------------
def compute_basic_metrics(G: nx.Graph, name: str) -> dict:
    """
    Compute basic network statistics.

    Returns a dict with:
    - name, nodes, edges, density, avg_degree, max_degree,
      num_components, largest_component_size
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()

    if n == 0:
        return {"name": name, "nodes": 0, "edges": 0, "density": 0,
                "avg_degree": 0, "max_degree": 0, "num_components": 0,
                "largest_component_size": 0}

    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees)
    max_degree = max(degrees)
    density = nx.density(G)

    # Connected components
    components = list(nx.connected_components(G))
    num_components = len(components)
    largest_component_size = max(len(c) for c in components) if components else 0

    metrics = {
        "name": name,
        "nodes": n,
        "edges": e,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 2),
        "max_degree": max_degree,
        "num_components": num_components,
        "largest_component_size": largest_component_size,
    }

    print(f"\n  [{name}] Basic Metrics:")
    for k, v in metrics.items():
        if k != "name":
            print(f"    {k}: {v}")

    return metrics


# ---------------------------------------------------------------------------
# Centrality Metrics
# ---------------------------------------------------------------------------
def compute_centrality_metrics(G: nx.Graph, name: str,
                                top_n: int = 20) -> pd.DataFrame:
    """
    Compute centrality metrics and return top-N nodes for each.

    Metrics: degree, betweenness, closeness, pagerank
    """
    print(f"\n  [{name}] Computing centrality metrics (top {top_n}) ...")

    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    # Compute centralities
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G, weight="weight")
    closeness_cent = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G, weight="weight")

    # Build a comprehensive DataFrame with all centrality values per node
    all_nodes = list(G.nodes())
    cent_df = pd.DataFrame({
        "node": all_nodes,
        "degree_centrality": [degree_cent.get(n, 0) for n in all_nodes],
        "betweenness_centrality": [betweenness_cent.get(n, 0) for n in all_nodes],
        "closeness_centrality": [closeness_cent.get(n, 0) for n in all_nodes],
        "pagerank": [pagerank.get(n, 0) for n in all_nodes],
        "degree": [G.degree(n) for n in all_nodes],
    })

    # Sort by degree centrality for the top-N view
    cent_df = cent_df.sort_values("degree_centrality", ascending=False)

    # Print top nodes
    print(f"    Top {top_n} by degree centrality:")
    for _, row in cent_df.head(top_n).iterrows():
        print(f"      {row['node']:20s}  deg={row['degree_centrality']:.4f}  "
              f"btwn={row['betweenness_centrality']:.4f}  "
              f"close={row['closeness_centrality']:.4f}  "
              f"pr={row['pagerank']:.4f}")

    return cent_df


# ---------------------------------------------------------------------------
# Clustering & Community Detection
# ---------------------------------------------------------------------------
def compute_clustering_and_communities(G: nx.Graph,
                                        name: str) -> tuple:
    """
    Compute clustering coefficient and Louvain community detection.

    Returns (community_dict, modularity, avg_clustering)
    - community_dict: {node: community_id}
    - modularity: float
    - avg_clustering: float
    """
    print(f"\n  [{name}] Computing clustering & communities ...")

    if G.number_of_nodes() == 0:
        return {}, 0.0, 0.0

    # Average clustering coefficient
    avg_clustering = nx.average_clustering(G)
    print(f"    Average clustering coefficient: {avg_clustering:.4f}")

    # Louvain community detection
    partition = community_louvain.best_partition(G, weight="weight",
                                                  random_state=42)
    modularity = community_louvain.modularity(partition, G, weight="weight")

    # Community size distribution
    from collections import Counter
    community_sizes = Counter(partition.values())
    num_communities = len(community_sizes)

    print(f"    Number of communities: {num_communities}")
    print(f"    Modularity score: {modularity:.4f}")
    print(f"    Community sizes: {dict(sorted(community_sizes.items()))}")

    return partition, modularity, avg_clustering


# ---------------------------------------------------------------------------
# Degree Distribution
# ---------------------------------------------------------------------------
def compute_degree_distribution(G: nx.Graph, name: str) -> pd.DataFrame:
    """
    Compute the degree distribution of the network.
    Returns a DataFrame with columns: degree, count, fraction
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["degree", "count", "fraction"])

    from collections import Counter
    degree_sequence = [d for _, d in G.degree()]
    degree_counts = Counter(degree_sequence)
    total = len(degree_sequence)

    dist_df = pd.DataFrame([
        {"degree": deg, "count": cnt, "fraction": cnt / total}
        for deg, cnt in sorted(degree_counts.items())
    ])

    print(f"\n  [{name}] Degree distribution:")
    print(f"    Min degree: {dist_df['degree'].min()}")
    print(f"    Max degree: {dist_df['degree'].max()}")
    print(f"    Median degree: {dist_df['degree'].median():.0f}")

    return dist_df


# ---------------------------------------------------------------------------
# Cross-Network Comparison
# ---------------------------------------------------------------------------
def compute_comparison(basic_dep: dict, basic_nondep: dict,
                       mod_dep: float, mod_nondep: float,
                       clust_dep: float, clust_nondep: float) -> pd.DataFrame:
    """
    Build a side-by-side comparison of depressive vs non-depressive networks.
    """
    print("\n  [comparison] Building side-by-side comparison ...")

    metrics_list = [
        "nodes", "edges", "density", "avg_degree", "max_degree",
        "num_components", "largest_component_size"
    ]

    comparison = []
    for m in metrics_list:
        comparison.append({
            "metric": m,
            "depressive": basic_dep.get(m, ""),
            "nondepressive": basic_nondep.get(m, ""),
        })

    # Add clustering and modularity
    comparison.append({
        "metric": "avg_clustering_coefficient",
        "depressive": round(clust_dep, 4),
        "nondepressive": round(clust_nondep, 4),
    })
    comparison.append({
        "metric": "modularity",
        "depressive": round(mod_dep, 4),
        "nondepressive": round(mod_nondep, 4),
    })

    return pd.DataFrame(comparison)


# ---------------------------------------------------------------------------
# Bridge Nodes & Exclusive Keywords
# ---------------------------------------------------------------------------
def find_bridge_nodes(G_combined: nx.Graph, top_n: int = 20) -> pd.DataFrame:
    """
    Identify bridge nodes in the combined network — keywords with high
    betweenness centrality that connect depressive and non-depressive
    communities.
    """
    print("\n  [bridge_nodes] Finding bridge nodes in combined network ...")

    if G_combined.number_of_nodes() == 0:
        return pd.DataFrame()

    betweenness = nx.betweenness_centrality(G_combined, weight="weight")

    bridge_data = []
    for node, btwn in sorted(betweenness.items(), key=lambda x: -x[1])[:top_n]:
        membership = G_combined.nodes[node].get("membership", "unknown")
        freq_dep = G_combined.nodes[node].get("freq_depressive", 0)
        freq_nondep = G_combined.nodes[node].get("freq_nondepressive", 0)
        bridge_data.append({
            "keyword": node,
            "betweenness_centrality": round(btwn, 6),
            "membership": membership,
            "freq_depressive": freq_dep,
            "freq_nondepressive": freq_nondep,
        })

    df = pd.DataFrame(bridge_data)
    print(f"    Top {top_n} bridge nodes:")
    for _, row in df.iterrows():
        print(f"      {row['keyword']:20s}  btwn={row['betweenness_centrality']:.4f}  "
              f"({row['membership']})")

    return df


def find_exclusive_keywords(G_combined: nx.Graph) -> pd.DataFrame:
    """
    Identify keywords that appear exclusively in depressive or
    non-depressive networks.
    """
    print("\n  [exclusive] Finding exclusive keywords ...")

    exclusive_data = []
    for node in G_combined.nodes():
        membership = G_combined.nodes[node].get("membership", "unknown")
        if membership in ("depressive_only", "nondepressive_only"):
            freq_dep = G_combined.nodes[node].get("freq_depressive", 0)
            freq_nondep = G_combined.nodes[node].get("freq_nondepressive", 0)
            exclusive_data.append({
                "keyword": node,
                "exclusive_to": membership.replace("_only", ""),
                "frequency": freq_dep if "depressive" in membership else freq_nondep,
            })

    df = pd.DataFrame(exclusive_data).sort_values("frequency", ascending=False)

    dep_only = df[df["exclusive_to"] == "depressive"]
    nondep_only = df[df["exclusive_to"] == "nondepressive"]
    print(f"    Depressive-only keywords: {len(dep_only)}")
    print(f"    Non-depressive-only keywords: {len(nondep_only)}")

    if len(dep_only) > 0:
        print(f"    Top depressive-only: {dep_only.head(10)['keyword'].tolist()}")
    if len(nondep_only) > 0:
        print(f"    Top nondepressive-only: {nondep_only.head(10)['keyword'].tolist()}")

    return df


# ---------------------------------------------------------------------------
# Save Results
# ---------------------------------------------------------------------------
def save_all_results(results: dict) -> None:
    """Save all computed metrics to CSV files."""
    ensure_dir(METRICS_DIR)

    for filename, df_or_dict in results.items():
        path = os.path.join(METRICS_DIR, filename)
        if isinstance(df_or_dict, pd.DataFrame):
            df_or_dict.to_csv(path, index=False)
        elif isinstance(df_or_dict, dict):
            pd.DataFrame([df_or_dict]).to_csv(path, index=False)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run all metric computations."""
    print("[metrics] Loading graphs ...")
    graphs = load_graphs()

    results = {}

    # ===================================================================
    # KEYWORD CO-OCCURRENCE NETWORKS
    # ===================================================================
    print("\n" + "=" * 60)
    print("KEYWORD CO-OCCURRENCE NETWORKS")
    print("=" * 60)

    G_kw_dep = graphs["keyword_depressive"]
    G_kw_nondep = graphs["keyword_nondepressive"]

    # Basic metrics
    basic_dep = compute_basic_metrics(G_kw_dep, "keyword_depressive")
    basic_nondep = compute_basic_metrics(G_kw_nondep, "keyword_nondepressive")
    results["basic_metrics.csv"] = pd.DataFrame([basic_dep, basic_nondep])

    # Centrality
    cent_dep = compute_centrality_metrics(G_kw_dep, "keyword_depressive")
    cent_nondep = compute_centrality_metrics(G_kw_nondep, "keyword_nondepressive")
    results["centrality_depressive.csv"] = cent_dep
    results["centrality_nondepressive.csv"] = cent_nondep

    # Clustering & Community
    part_dep, mod_dep, clust_dep = compute_clustering_and_communities(
        G_kw_dep, "keyword_depressive")
    part_nondep, mod_nondep, clust_nondep = compute_clustering_and_communities(
        G_kw_nondep, "keyword_nondepressive")

    # Save community assignments
    if part_dep:
        comm_dep = pd.DataFrame([
            {"node": node, "community": comm, "network": "depressive"}
            for node, comm in part_dep.items()
        ])
        comm_nondep = pd.DataFrame([
            {"node": node, "community": comm, "network": "nondepressive"}
            for node, comm in part_nondep.items()
        ])
        results["communities.csv"] = pd.concat([comm_dep, comm_nondep],
                                                ignore_index=True)

    # Degree distributions
    dist_dep = compute_degree_distribution(G_kw_dep, "keyword_depressive")
    dist_nondep = compute_degree_distribution(G_kw_nondep, "keyword_nondepressive")
    dist_dep["network"] = "depressive"
    dist_nondep["network"] = "nondepressive"
    results["degree_distribution.csv"] = pd.concat([dist_dep, dist_nondep],
                                                    ignore_index=True)

    # ===================================================================
    # TWEET SIMILARITY NETWORKS
    # ===================================================================
    print("\n" + "=" * 60)
    print("TWEET SIMILARITY NETWORKS")
    print("=" * 60)

    G_tw_dep = graphs["tweet_similarity_depressive"]
    G_tw_nondep = graphs["tweet_similarity_nondepressive"]

    basic_tw_dep = compute_basic_metrics(G_tw_dep, "tweet_sim_depressive")
    basic_tw_nondep = compute_basic_metrics(G_tw_nondep, "tweet_sim_nondepressive")

    # Append to basic metrics
    basic_all = results["basic_metrics.csv"]
    basic_all = pd.concat([basic_all, pd.DataFrame([basic_tw_dep, basic_tw_nondep])],
                          ignore_index=True)

    # Tweet similarity community detection
    part_tw_dep, mod_tw_dep, clust_tw_dep = compute_clustering_and_communities(
        G_tw_dep, "tweet_sim_depressive")
    part_tw_nondep, mod_tw_nondep, clust_tw_nondep = compute_clustering_and_communities(
        G_tw_nondep, "tweet_sim_nondepressive")

    # ===================================================================
    # COMBINED KEYWORD NETWORK
    # ===================================================================
    print("\n" + "=" * 60)
    print("COMBINED KEYWORD NETWORK")
    print("=" * 60)

    G_combined = graphs["combined_keyword"]

    basic_combined = compute_basic_metrics(G_combined, "combined_keyword")
    basic_all = pd.concat([basic_all, pd.DataFrame([basic_combined])],
                          ignore_index=True)
    results["basic_metrics.csv"] = basic_all

    cent_combined = compute_centrality_metrics(G_combined, "combined_keyword")
    results["centrality_combined.csv"] = cent_combined

    part_combined, mod_combined, clust_combined = compute_clustering_and_communities(
        G_combined, "combined_keyword")

    # Bridge nodes
    bridge_df = find_bridge_nodes(G_combined)
    results["bridge_nodes.csv"] = bridge_df

    # Exclusive keywords
    exclusive_df = find_exclusive_keywords(G_combined)
    results["exclusive_keywords.csv"] = exclusive_df

    # ===================================================================
    # COMPARISON: Depressive vs Non-depressive (keyword networks)
    # ===================================================================
    print("\n" + "=" * 60)
    print("COMPARISON: DEPRESSIVE vs NON-DEPRESSIVE")
    print("=" * 60)

    comparison_df = compute_comparison(basic_dep, basic_nondep,
                                       mod_dep, mod_nondep,
                                       clust_dep, clust_nondep)
    results["comparison.csv"] = comparison_df

    # Print comparison
    print("\n  Side-by-side comparison:")
    for _, row in comparison_df.iterrows():
        print(f"    {row['metric']:30s}  dep={row['depressive']}  "
              f"nondep={row['nondepressive']}")

    # ===================================================================
    # ATTACH COMMUNITY IDs TO GRAPHS (for Gephi export later)
    # ===================================================================
    print("\n[metrics] Attaching community IDs to graph nodes ...")

    # Keyword networks
    for node, comm in part_dep.items():
        G_kw_dep.nodes[node]["community"] = comm
    for node, comm in part_nondep.items():
        G_kw_nondep.nodes[node]["community"] = comm
    for node, comm in part_combined.items():
        G_combined.nodes[node]["community"] = comm

    # Tweet similarity networks
    for node, comm in part_tw_dep.items():
        G_tw_dep.nodes[node]["community"] = comm
    for node, comm in part_tw_nondep.items():
        G_tw_nondep.nodes[node]["community"] = comm

    # Attach centrality scores to keyword network nodes
    for _, row in cent_dep.iterrows():
        node = row["node"]
        if node in G_kw_dep.nodes:
            G_kw_dep.nodes[node]["degree_centrality"] = round(row["degree_centrality"], 4)
            G_kw_dep.nodes[node]["betweenness_centrality"] = round(row["betweenness_centrality"], 4)
            G_kw_dep.nodes[node]["closeness_centrality"] = round(row["closeness_centrality"], 4)
            G_kw_dep.nodes[node]["pagerank"] = round(row["pagerank"], 4)

    for _, row in cent_nondep.iterrows():
        node = row["node"]
        if node in G_kw_nondep.nodes:
            G_kw_nondep.nodes[node]["degree_centrality"] = round(row["degree_centrality"], 4)
            G_kw_nondep.nodes[node]["betweenness_centrality"] = round(row["betweenness_centrality"], 4)
            G_kw_nondep.nodes[node]["closeness_centrality"] = round(row["closeness_centrality"], 4)
            G_kw_nondep.nodes[node]["pagerank"] = round(row["pagerank"], 4)

    for _, row in cent_combined.iterrows():
        node = row["node"]
        if node in G_combined.nodes:
            G_combined.nodes[node]["degree_centrality"] = round(row["degree_centrality"], 4)
            G_combined.nodes[node]["betweenness_centrality"] = round(row["betweenness_centrality"], 4)
            G_combined.nodes[node]["closeness_centrality"] = round(row["closeness_centrality"], 4)
            G_combined.nodes[node]["pagerank"] = round(row["pagerank"], 4)

    # Re-save graphs with updated attributes
    from src.build_graph import save_graphs
    save_graphs(graphs)

    # ===================================================================
    # SAVE ALL METRICS
    # ===================================================================
    print("\n[metrics] Saving all metric results ...")
    save_all_results(results)

    print("\n[metrics] Done. Run `python -m src.export` next.")


if __name__ == "__main__":
    main()
