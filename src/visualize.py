"""
visualize.py — Degree Distribution Plots & Metric Comparison Charts
=====================================================================
Generates matplotlib/seaborn visualizations for the SNA analysis:
- Degree distribution histograms for keyword networks
- Degree distribution for tweet similarity networks
- Side-by-side metric comparison bar chart

Usage:
    python -m src.visualize
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script use
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from src.build_graph import load_graphs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METRICS_DIR = os.path.join("outputs", "metrics")
PLOTS_DIR = os.path.join("outputs", "metrics", "plots")

# Style settings
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Degree Distribution Plots
# ---------------------------------------------------------------------------
def plot_degree_distribution(G: nx.Graph, title: str, filename: str,
                              color: str = "#3498db") -> None:
    """
    Plot the degree distribution histogram for a graph.
    """
    degrees = [d for _, d in G.degree()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(degrees, bins=30, color=color, edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Degree Distribution — {title}")
    axes[0].axvline(x=sum(degrees) / len(degrees), color="red",
                     linestyle="--", label=f"Mean: {sum(degrees)/len(degrees):.1f}")
    axes[0].legend()

    # Log-log plot (for power-law check)
    from collections import Counter
    degree_counts = Counter(degrees)
    deg_vals = sorted(degree_counts.keys())
    deg_freqs = [degree_counts[d] for d in deg_vals]

    axes[1].scatter(deg_vals, deg_freqs, color=color, alpha=0.7, s=30)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Degree (log)")
    axes[1].set_ylabel("Frequency (log)")
    axes[1].set_title(f"Log-Log Degree Distribution — {title}")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_degree_comparison(G_dep: nx.Graph, G_nondep: nx.Graph,
                            title: str, filename: str) -> None:
    """
    Plot overlaid degree distributions for depressive vs non-depressive.
    """
    deg_dep = [d for _, d in G_dep.degree()]
    deg_nondep = [d for _, d in G_nondep.degree()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(deg_dep, bins=30, alpha=0.6, color="#e74c3c", label="Depressive (1)",
            edgecolor="white")
    ax.hist(deg_nondep, bins=30, alpha=0.6, color="#2ecc71",
            label="Non-depressive (0)", edgecolor="white")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Metric Comparison Bar Chart
# ---------------------------------------------------------------------------
def plot_metric_comparison(comparison_path: str, filename: str) -> None:
    """
    Plot a bar chart comparing depressive vs non-depressive metrics.
    """
    df = pd.read_csv(comparison_path)

    # Select numeric-friendly metrics for bar chart
    # Skip 'nodes' and 'edges' (same for both) and focus on meaningful diffs
    chart_metrics = ["density", "avg_degree", "avg_clustering_coefficient",
                     "modularity"]

    chart_data = df[df["metric"].isin(chart_metrics)].copy()
    chart_data["depressive"] = pd.to_numeric(chart_data["depressive"])
    chart_data["nondepressive"] = pd.to_numeric(chart_data["nondepressive"])

    # Reshape for grouped bar chart
    x = range(len(chart_data))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar([i - width / 2 for i in x],
                    chart_data["depressive"].values,
                    width, label="Depressive (1)", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar([i + width / 2 for i in x],
                    chart_data["nondepressive"].values,
                    width, label="Non-depressive (0)", color="#2ecc71", alpha=0.8)

    ax.set_ylabel("Value")
    ax.set_title("SNA Metric Comparison: Depressive vs Non-Depressive Networks")
    ax.set_xticks(list(x))
    ax.set_xticklabels(chart_data["metric"].values, rotation=15, ha="right")
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Top Centrality Bar Chart
# ---------------------------------------------------------------------------
def plot_top_centrality(cent_path: str, metric: str, title: str,
                         filename: str, top_n: int = 15,
                         color: str = "#3498db") -> None:
    """
    Plot a horizontal bar chart for top-N nodes by a given centrality metric.
    """
    df = pd.read_csv(cent_path)
    df = df.sort_values(metric, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(df)), df[metric].values, color=color, alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["node"].values)
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Generate all visualization plots."""
    ensure_dir(PLOTS_DIR)

    print("[visualize] Loading graphs ...")
    graphs = load_graphs()

    G_kw_dep = graphs["keyword_depressive"]
    G_kw_nondep = graphs["keyword_nondepressive"]
    G_tw_dep = graphs["tweet_similarity_depressive"]
    G_tw_nondep = graphs["tweet_similarity_nondepressive"]

    # ---- Degree Distribution Plots ----
    print("\n[visualize] Generating degree distribution plots ...")

    plot_degree_distribution(G_kw_dep, "Keyword Network (Depressive)",
                              "degree_dist_keyword_depressive.png",
                              color="#e74c3c")
    plot_degree_distribution(G_kw_nondep, "Keyword Network (Non-Depressive)",
                              "degree_dist_keyword_nondepressive.png",
                              color="#2ecc71")
    plot_degree_distribution(G_tw_dep, "Tweet Similarity (Depressive)",
                              "degree_dist_tweet_depressive.png",
                              color="#e74c3c")
    plot_degree_distribution(G_tw_nondep, "Tweet Similarity (Non-Depressive)",
                              "degree_dist_tweet_nondepressive.png",
                              color="#2ecc71")

    # Overlaid comparisons
    plot_degree_comparison(G_kw_dep, G_kw_nondep,
                           "Keyword Network Degree Distribution Comparison",
                           "degree_comparison_keyword.png")
    plot_degree_comparison(G_tw_dep, G_tw_nondep,
                           "Tweet Similarity Degree Distribution Comparison",
                           "degree_comparison_tweet.png")

    # ---- Metric Comparison ----
    print("\n[visualize] Generating metric comparison chart ...")
    comparison_path = os.path.join(METRICS_DIR, "comparison.csv")
    plot_metric_comparison(comparison_path, "metric_comparison.png")

    # ---- Top Centrality Plots ----
    print("\n[visualize] Generating top centrality bar charts ...")

    cent_dep_path = os.path.join(METRICS_DIR, "centrality_depressive.csv")
    cent_nondep_path = os.path.join(METRICS_DIR, "centrality_nondepressive.csv")

    for metric in ["degree_centrality", "betweenness_centrality", "pagerank"]:
        plot_top_centrality(
            cent_dep_path, metric,
            f"Top 15 Keywords by {metric.replace('_', ' ').title()} (Depressive)",
            f"top_{metric}_depressive.png",
            color="#e74c3c"
        )
        plot_top_centrality(
            cent_nondep_path, metric,
            f"Top 15 Keywords by {metric.replace('_', ' ').title()} (Non-Depressive)",
            f"top_{metric}_nondepressive.png",
            color="#2ecc71"
        )

    print(f"\n[visualize] All plots saved to {PLOTS_DIR}/")
    print("[visualize] Done.")


if __name__ == "__main__":
    main()
