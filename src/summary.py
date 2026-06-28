"""
summary.py — Generate Summary CSV with All SNA Metrics
========================================================
Combines all computed metrics into a single comprehensive summary CSV
for use in the written academic report.

Usage:
    python -m src.summary
"""

import os
import pandas as pd
import networkx as nx
import community as community_louvain

from src.build_graph import load_graphs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METRICS_DIR = os.path.join("outputs", "metrics")


def generate_summary() -> pd.DataFrame:
    """
    Generate a comprehensive summary CSV with all SNA metrics
    side-by-side for depressive vs non-depressive networks.
    """
    print("[summary] Loading graphs and metrics ...")
    graphs = load_graphs()

    G_kw_dep = graphs["keyword_depressive"]
    G_kw_nondep = graphs["keyword_nondepressive"]
    G_tw_dep = graphs["tweet_similarity_depressive"]
    G_tw_nondep = graphs["tweet_similarity_nondepressive"]
    G_combined = graphs["combined_keyword"]

    rows = []

    def add_row(category, metric, dep_val, nondep_val, combined_val="—"):
        rows.append({
            "category": category,
            "metric": metric,
            "depressive": dep_val,
            "nondepressive": nondep_val,
            "combined": combined_val,
        })

    # ---- Keyword Co-occurrence Network Metrics ----
    cat = "Keyword Network"

    add_row(cat, "Nodes", G_kw_dep.number_of_nodes(),
            G_kw_nondep.number_of_nodes(), G_combined.number_of_nodes())
    add_row(cat, "Edges", G_kw_dep.number_of_edges(),
            G_kw_nondep.number_of_edges(), G_combined.number_of_edges())
    add_row(cat, "Density", round(nx.density(G_kw_dep), 4),
            round(nx.density(G_kw_nondep), 4), round(nx.density(G_combined), 4))

    deg_dep = [d for _, d in G_kw_dep.degree()]
    deg_nondep = [d for _, d in G_kw_nondep.degree()]
    add_row(cat, "Avg Degree", round(sum(deg_dep) / len(deg_dep), 2),
            round(sum(deg_nondep) / len(deg_nondep), 2))
    add_row(cat, "Max Degree", max(deg_dep), max(deg_nondep))

    add_row(cat, "Avg Clustering Coeff",
            round(nx.average_clustering(G_kw_dep), 4),
            round(nx.average_clustering(G_kw_nondep), 4),
            round(nx.average_clustering(G_combined), 4))

    add_row(cat, "Connected Components",
            nx.number_connected_components(G_kw_dep),
            nx.number_connected_components(G_kw_nondep),
            nx.number_connected_components(G_combined))

    # Louvain modularity
    part_dep = community_louvain.best_partition(G_kw_dep, weight="weight",
                                                 random_state=42)
    part_nondep = community_louvain.best_partition(G_kw_nondep, weight="weight",
                                                    random_state=42)
    part_combined = community_louvain.best_partition(G_combined, weight="weight",
                                                      random_state=42)

    mod_dep = community_louvain.modularity(part_dep, G_kw_dep, weight="weight")
    mod_nondep = community_louvain.modularity(part_nondep, G_kw_nondep,
                                               weight="weight")
    mod_combined = community_louvain.modularity(part_combined, G_combined,
                                                 weight="weight")

    add_row(cat, "Communities (Louvain)",
            len(set(part_dep.values())),
            len(set(part_nondep.values())),
            len(set(part_combined.values())))
    add_row(cat, "Modularity Score",
            round(mod_dep, 4), round(mod_nondep, 4), round(mod_combined, 4))

    # Top centrality nodes
    dc_dep = nx.degree_centrality(G_kw_dep)
    dc_nondep = nx.degree_centrality(G_kw_nondep)
    top_dep = sorted(dc_dep, key=dc_dep.get, reverse=True)[:5]
    top_nondep = sorted(dc_nondep, key=dc_nondep.get, reverse=True)[:5]
    add_row(cat, "Top-5 Degree Centrality Nodes",
            ", ".join(top_dep), ", ".join(top_nondep))

    bc_dep = nx.betweenness_centrality(G_kw_dep, weight="weight")
    bc_nondep = nx.betweenness_centrality(G_kw_nondep, weight="weight")
    top_bc_dep = sorted(bc_dep, key=bc_dep.get, reverse=True)[:5]
    top_bc_nondep = sorted(bc_nondep, key=bc_nondep.get, reverse=True)[:5]
    add_row(cat, "Top-5 Betweenness Centrality Nodes",
            ", ".join(top_bc_dep), ", ".join(top_bc_nondep))

    pr_dep = nx.pagerank(G_kw_dep, weight="weight")
    pr_nondep = nx.pagerank(G_kw_nondep, weight="weight")
    top_pr_dep = sorted(pr_dep, key=pr_dep.get, reverse=True)[:5]
    top_pr_nondep = sorted(pr_nondep, key=pr_nondep.get, reverse=True)[:5]
    add_row(cat, "Top-5 PageRank Nodes",
            ", ".join(top_pr_dep), ", ".join(top_pr_nondep))

    # ---- Tweet Similarity Network Metrics ----
    cat = "Tweet Similarity Network"

    add_row(cat, "Nodes (sampled)", G_tw_dep.number_of_nodes(),
            G_tw_nondep.number_of_nodes())
    add_row(cat, "Edges", G_tw_dep.number_of_edges(),
            G_tw_nondep.number_of_edges())
    add_row(cat, "Density", round(nx.density(G_tw_dep), 6),
            round(nx.density(G_tw_nondep), 6))

    tw_deg_dep = [d for _, d in G_tw_dep.degree()]
    tw_deg_nondep = [d for _, d in G_tw_nondep.degree()]
    add_row(cat, "Avg Degree",
            round(sum(tw_deg_dep) / len(tw_deg_dep), 2),
            round(sum(tw_deg_nondep) / len(tw_deg_nondep), 2))

    add_row(cat, "Avg Clustering Coeff",
            round(nx.average_clustering(G_tw_dep), 4),
            round(nx.average_clustering(G_tw_nondep), 4))

    add_row(cat, "Connected Components",
            nx.number_connected_components(G_tw_dep),
            nx.number_connected_components(G_tw_nondep))

    tw_part_dep = community_louvain.best_partition(G_tw_dep, random_state=42)
    tw_part_nondep = community_louvain.best_partition(G_tw_nondep, random_state=42)
    tw_mod_dep = community_louvain.modularity(tw_part_dep, G_tw_dep)
    tw_mod_nondep = community_louvain.modularity(tw_part_nondep, G_tw_nondep)

    add_row(cat, "Communities (Louvain)",
            len(set(tw_part_dep.values())),
            len(set(tw_part_nondep.values())))
    add_row(cat, "Modularity Score",
            round(tw_mod_dep, 4), round(tw_mod_nondep, 4))

    # ---- Combined Network: Bridge & Exclusive Info ----
    cat = "Cross-Network Analysis"

    # Membership counts
    memberships = nx.get_node_attributes(G_combined, "membership")
    from collections import Counter
    mem_counts = Counter(memberships.values())

    add_row(cat, "Shared Keywords",
            mem_counts.get("both", 0), mem_counts.get("both", 0),
            mem_counts.get("both", 0))
    add_row(cat, "Exclusive Keywords",
            mem_counts.get("depressive_only", 0),
            mem_counts.get("nondepressive_only", 0))

    # Top bridge nodes
    bc_combined = nx.betweenness_centrality(G_combined, weight="weight")
    top_bridges = sorted(bc_combined, key=bc_combined.get, reverse=True)[:5]
    add_row(cat, "Top-5 Bridge Nodes (Betweenness)",
            ", ".join(top_bridges), ", ".join(top_bridges),
            ", ".join(top_bridges))

    # Build DataFrame
    summary_df = pd.DataFrame(rows)
    return summary_df


def main():
    """Generate and save the summary CSV."""
    summary_df = generate_summary()

    # Save
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, "summary.csv")
    summary_df.to_csv(path, index=False)

    print(f"\n[summary] Summary saved to {path}")
    print(f"[summary] {len(summary_df)} rows of metrics\n")

    # Print the summary
    print("=" * 80)
    print("COMPLETE SNA SUMMARY")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        if row["category"] != (summary_df.iloc[_ - 1]["category"]
                                if _ > 0 else ""):
            print(f"\n--- {row['category']} ---")
        print(f"  {row['metric']:40s}  "
              f"dep={str(row['depressive']):>20s}  "
              f"nondep={str(row['nondepressive']):>20s}")
    print("=" * 80)


if __name__ == "__main__":
    main()
