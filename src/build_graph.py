"""
build_graph.py — Network Construction
=======================================
Builds three types of networks from the preprocessed tweet data:

Graph A: Keyword Co-occurrence Network (depressive & non-depressive separately)
Graph B: Tweet Similarity Network (sampled, depressive & non-depressive separately)
Graph C: Combined Keyword Network (merged, with category tags)

Usage:
    python -m src.build_graph
"""

import os
import pickle
import pandas as pd
import networkx as nx
from itertools import combinations
from collections import Counter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join("data", "processed")
GRAPH_DIR = os.path.join("outputs", "graphs")

# Number of top keywords to use for the co-occurrence network
TOP_N_KEYWORDS = 200

# Tweet similarity network settings
TWEET_SAMPLE_SIZE = 2500       # tweets per category to sample
JACCARD_THRESHOLD = 0.15       # minimum Jaccard similarity to create an edge


def load_processed_data() -> tuple:
    """
    Load the processed dataset and split into depressive / non-depressive.
    Returns (dep_df, nondep_df) with 'keywords' as lists.
    """
    print("[build_graph] Loading processed data ...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "all_tweets_processed.csv"))

    # Reconstruct keyword lists from pipe-separated strings
    df["keywords"] = df["keywords"].apply(
        lambda x: x.split("|") if isinstance(x, str) and x else []
    )
    df = df[df["keywords"].apply(len) > 0]

    dep = df[df["sentiment"] == 1].copy()
    nondep = df[df["sentiment"] == 0].copy()

    print(f"[build_graph] Depressive tweets: {len(dep)}")
    print(f"[build_graph] Non-depressive tweets: {len(nondep)}")
    return dep, nondep


def get_top_keywords_from_file(sentiment_label: str) -> set:
    """Load the top keyword list for a given sentiment category."""
    filename = f"top_keywords_{sentiment_label}.csv"
    path = os.path.join(PROCESSED_DIR, filename)
    kw_df = pd.read_csv(path)
    return set(kw_df["keyword"].tolist())


# ---------------------------------------------------------------------------
# Graph A: Keyword Co-occurrence Network
# ---------------------------------------------------------------------------
def build_keyword_cooccurrence_graph(df: pd.DataFrame,
                                     top_keywords: set,
                                     label: str) -> nx.Graph:
    """
    Build a keyword co-occurrence network.

    Nodes = keywords (filtered to top_keywords set)
    Edge  = two keywords co-occur in the same tweet
    Weight = number of tweets in which they co-occur

    Parameters
    ----------
    df : DataFrame with 'keywords' column (list of strings)
    top_keywords : set of keywords to include as nodes
    label : descriptive label for logging

    Returns
    -------
    nx.Graph with node attribute 'category' and edge attribute 'weight'
    """
    print(f"\n[build_graph] Building keyword co-occurrence network: {label}")

    # Count co-occurrences
    cooccurrence = Counter()
    keyword_freq = Counter()

    for kw_list in df["keywords"]:
        # Filter to top keywords only
        filtered = [kw for kw in kw_list if kw in top_keywords]
        # Remove duplicates within a single tweet
        unique_kw = list(set(filtered))

        # Count individual keyword frequencies
        for kw in unique_kw:
            keyword_freq[kw] += 1

        # Count co-occurrences (all pairs)
        for pair in combinations(sorted(unique_kw), 2):
            cooccurrence[pair] += 1

    # Build graph
    G = nx.Graph()

    # Add nodes with frequency attribute
    for kw, freq in keyword_freq.items():
        G.add_node(kw, frequency=freq, category=label)

    # Add edges with weight
    for (kw1, kw2), weight in cooccurrence.items():
        if weight >= 2:  # Filter out very rare co-occurrences
            G.add_edge(kw1, kw2, weight=weight)

    # Remove isolated nodes (no co-occurrences)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Isolated nodes removed: {len(isolated)}")

    return G


# ---------------------------------------------------------------------------
# Graph B: Tweet Similarity Network
# ---------------------------------------------------------------------------
def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def build_tweet_similarity_graph(df: pd.DataFrame,
                                  label: str,
                                  sample_size: int = TWEET_SAMPLE_SIZE,
                                  threshold: float = JACCARD_THRESHOLD) -> nx.Graph:
    """
    Build a tweet similarity network from sampled tweets.

    Nodes = tweets (identified by index)
    Edge  = Jaccard similarity of keyword sets > threshold
    Weight = Jaccard similarity value

    Parameters
    ----------
    df : DataFrame with 'keywords' column
    label : 'depressive' or 'nondepressive'
    sample_size : number of tweets to sample
    threshold : minimum Jaccard similarity for an edge

    Returns
    -------
    nx.Graph
    """
    print(f"\n[build_graph] Building tweet similarity network: {label}")
    print(f"  Sampling {sample_size} tweets (threshold={threshold}) ...")

    # Sample tweets
    if len(df) > sample_size:
        sampled = df.sample(n=sample_size, random_state=42)
    else:
        sampled = df

    # Prepare keyword sets
    tweet_data = []
    for idx, row in sampled.iterrows():
        kw_set = set(row["keywords"])
        if len(kw_set) >= 2:  # Need at least 2 keywords for meaningful similarity
            tweet_data.append((idx, kw_set, row.get("clean_text", "")[:50]))

    print(f"  Tweets with ≥2 keywords: {len(tweet_data)}")

    # Build graph
    G = nx.Graph()

    # Add all nodes
    for idx, kw_set, snippet in tweet_data:
        G.add_node(idx, category=label, keyword_count=len(kw_set),
                   snippet=snippet)

    # Compute pairwise similarities
    edge_count = 0
    total_pairs = len(tweet_data) * (len(tweet_data) - 1) // 2
    print(f"  Computing {total_pairs:,} pairwise similarities ...")

    for i in range(len(tweet_data)):
        for j in range(i + 1, len(tweet_data)):
            sim = jaccard_similarity(tweet_data[i][1], tweet_data[j][1])
            if sim >= threshold:
                G.add_edge(tweet_data[i][0], tweet_data[j][0], weight=round(sim, 4))
                edge_count += 1

    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {edge_count}")

    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    print(f"  After removing isolates: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    return G


# ---------------------------------------------------------------------------
# Graph C: Combined Keyword Network
# ---------------------------------------------------------------------------
def build_combined_keyword_graph(G_dep: nx.Graph,
                                  G_nondep: nx.Graph) -> nx.Graph:
    """
    Merge depressive and non-depressive keyword co-occurrence networks.

    Each node is tagged with its membership:
    - 'depressive_only'
    - 'nondepressive_only'
    - 'both'

    Edge weights are summed where both networks share an edge.
    """
    print("\n[build_graph] Building combined keyword network ...")

    G_combined = nx.Graph()

    dep_nodes = set(G_dep.nodes())
    nondep_nodes = set(G_nondep.nodes())
    all_nodes = dep_nodes | nondep_nodes

    # Classify and add nodes
    for node in all_nodes:
        in_dep = node in dep_nodes
        in_nondep = node in nondep_nodes

        if in_dep and in_nondep:
            membership = "both"
        elif in_dep:
            membership = "depressive_only"
        else:
            membership = "nondepressive_only"

        # Merge frequency attributes
        freq_dep = G_dep.nodes[node].get("frequency", 0) if in_dep else 0
        freq_nondep = G_nondep.nodes[node].get("frequency", 0) if in_nondep else 0

        G_combined.add_node(node,
                            membership=membership,
                            freq_depressive=freq_dep,
                            freq_nondepressive=freq_nondep,
                            freq_total=freq_dep + freq_nondep)

    # Merge edges
    all_edges = {}

    for u, v, data in G_dep.edges(data=True):
        key = tuple(sorted([u, v]))
        all_edges[key] = {"weight_dep": data.get("weight", 1),
                          "weight_nondep": 0}

    for u, v, data in G_nondep.edges(data=True):
        key = tuple(sorted([u, v]))
        if key in all_edges:
            all_edges[key]["weight_nondep"] = data.get("weight", 1)
        else:
            all_edges[key] = {"weight_dep": 0,
                              "weight_nondep": data.get("weight", 1)}

    for (u, v), weights in all_edges.items():
        total_weight = weights["weight_dep"] + weights["weight_nondep"]
        G_combined.add_edge(u, v,
                            weight=total_weight,
                            weight_depressive=weights["weight_dep"],
                            weight_nondepressive=weights["weight_nondep"])

    print(f"  Nodes: {G_combined.number_of_nodes()}")
    print(f"  Edges: {G_combined.number_of_edges()}")

    # Membership breakdown
    memberships = Counter(nx.get_node_attributes(G_combined, "membership").values())
    for m, count in sorted(memberships.items()):
        print(f"    {m}: {count} nodes")

    return G_combined


# ---------------------------------------------------------------------------
# Persistence — save / load graphs
# ---------------------------------------------------------------------------
def save_graphs(graphs: dict) -> None:
    """Save all graph objects to pickle for downstream use."""
    os.makedirs(GRAPH_DIR, exist_ok=True)
    pickle_path = os.path.join(GRAPH_DIR, "all_graphs.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"\n[build_graph] Saved {len(graphs)} graphs to {pickle_path}")


def load_graphs() -> dict:
    """Load all graph objects from pickle."""
    pickle_path = os.path.join(GRAPH_DIR, "all_graphs.pkl")
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Build all three network types."""
    dep, nondep = load_processed_data()

    # Load top keywords
    top_kw_dep = get_top_keywords_from_file("depressive")
    top_kw_nondep = get_top_keywords_from_file("nondepressive")

    # ---- Graph A: Keyword Co-occurrence Networks ----
    G_kw_dep = build_keyword_cooccurrence_graph(dep, top_kw_dep, "depressive")
    G_kw_nondep = build_keyword_cooccurrence_graph(nondep, top_kw_nondep,
                                                    "nondepressive")

    # ---- Graph B: Tweet Similarity Networks ----
    G_tweet_dep = build_tweet_similarity_graph(dep, "depressive")
    G_tweet_nondep = build_tweet_similarity_graph(nondep, "nondepressive")

    # ---- Graph C: Combined Keyword Network ----
    G_combined = build_combined_keyword_graph(G_kw_dep, G_kw_nondep)

    # Save all graphs
    graphs = {
        "keyword_depressive": G_kw_dep,
        "keyword_nondepressive": G_kw_nondep,
        "tweet_similarity_depressive": G_tweet_dep,
        "tweet_similarity_nondepressive": G_tweet_nondep,
        "combined_keyword": G_combined,
    }
    save_graphs(graphs)

    print("\n[build_graph] All networks constructed successfully.")
    print("[build_graph] Run `python -m src.metrics` next.")


if __name__ == "__main__":
    main()
