# Social Network Analysis of Depressive vs Non-Depressive Tweets

## Academic Context

This project applies **Social Network Analysis (SNA)** techniques to a corpus of
pre-labeled Twitter data (Dec 2019 – Dec 2020) from the IEEE DataPort. The goal is
to understand how the *structural and relational properties* of depressive
conversations differ from non-depressive ones at the network level — using graph
theory and SNA metrics only. No machine learning or predictive modeling is used.

## Dataset

**Source:** IEEE DataPort — *"Depressive/Non-Depressive Tweets Between Dec'19 to Dec'20"*

- 134,348 tweets labeled as depressive (`1`) or non-depressive (`0`)
- Pre-cleaned text (lowercased, URLs/mentions/hashtags removed by the data provider)
- Balanced distribution: ~68.7K depressive, ~65.7K non-depressive

> **Citation:** Place the original dataset in `data/raw/clean_tweet_Dec19ToDec20.csv`.
> Do **not** modify raw data files.

## Project Structure

```
sna-depression-tweets/
├── data/
│   └── raw/                  ← original dataset (never modified)
├── notebooks/                ← exploratory work
├── src/
│   ├── preprocess.py         ← data loading, cleaning, keyword extraction
│   ├── build_graph.py        ← network construction (3 graph types)
│   ├── metrics.py            ← all SNA metric computations
│   ├── export.py             ← export to Gephi-compatible formats
│   └── visualize.py          ← degree distribution & comparison plots
├── outputs/
│   ├── graphs/               ← .gexf and .graphml exports
│   └── metrics/              ← CSV exports of computed metrics
├── report/                   ← final write-up and screenshots
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/SOUMYA0023/Social-Network-Analysis.git
cd Social-Network-Analysis/sna-depression-tweets

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## How to Run

Execute the scripts **in order** from the project root (`sna-depression-tweets/`):

```bash
# Step 1: Load and preprocess the data, extract keywords
python -m src.preprocess

# Step 2: Build all three network graphs
python -m src.build_graph

# Step 3: Compute SNA metrics for all graphs
python -m src.metrics

# Step 4: Export graphs to Gephi-compatible formats (.gexf / .graphml)
python -m src.export

# Step 5: Generate degree distribution plots and comparison charts
python -m src.visualize
```

## Output Files

### `outputs/graphs/`
| File | Description |
|------|-------------|
| `keyword_depressive.gexf` / `.graphml` | Keyword co-occurrence network for depressive tweets |
| `keyword_nondepressive.gexf` / `.graphml` | Keyword co-occurrence network for non-depressive tweets |
| `tweet_similarity_depressive.gexf` / `.graphml` | Tweet similarity network (depressive subset, sampled) |
| `tweet_similarity_nondepressive.gexf` / `.graphml` | Tweet similarity network (non-depressive subset, sampled) |
| `combined_keyword.gexf` / `.graphml` | Merged keyword network with category tags |

### `outputs/metrics/`
| File | Description |
|------|-------------|
| `basic_metrics.csv` | Node/edge counts, density, avg degree for all graphs |
| `centrality_depressive.csv` | Top-20 centrality scores (depressive keyword network) |
| `centrality_nondepressive.csv` | Top-20 centrality scores (non-depressive keyword network) |
| `communities.csv` | Community assignments and modularity scores |
| `comparison.csv` | Side-by-side metric comparison (depressive vs non-depressive) |
| `bridge_nodes.csv` | High-betweenness nodes connecting communities |
| `exclusive_keywords.csv` | Keywords unique to depressive / non-depressive networks |
| `summary.csv` | Complete summary of all SNA metrics |

### `outputs/metrics/plots/`
| File | Description |
|------|-------------|
| `degree_distribution_*.png` | Degree distribution histograms |
| `metric_comparison.png` | Bar chart comparing key metrics |

## SNA Metrics Computed

### Basic Network Statistics
- **Node/Edge count**: Network size and connectivity
- **Density**: Proportion of possible edges that exist (sparse vs dense)
- **Average degree**: Mean connections per node

### Centrality Measures
- **Degree centrality**: Most connected keywords — topical hubs
- **Betweenness centrality**: Keywords that bridge different conversational clusters
- **Closeness centrality**: Keywords closest to all others — core discourse terms
- **PageRank**: Importance weighted by neighbor importance (recursive influence)

### Clustering & Community
- **Clustering coefficient**: How tightly neighbors are interconnected
- **Connected components**: Fragmentation of the network
- **Louvain community detection**: Identifies topic clusters within each network
- **Modularity**: Strength of community structure (higher = more distinct clusters)

### Cross-Network Comparison
- All metrics compared side-by-side for depressive vs non-depressive networks
- **Exclusive keywords**: Terms appearing only in one category
- **Bridge nodes**: High-betweenness keywords connecting depressive and non-depressive discourse

## Why These Metrics?

These metrics reveal how depressive discourse *structurally differs* from
non-depressive discourse. For example:
- Higher **density** in depressive networks may indicate more repetitive, circular language
- Different **community structures** reveal distinct topic clusters in each category
- **Bridge nodes** identify keywords that connect depressive and non-depressive conversations
- **Exclusive keywords** highlight vocabulary unique to each emotional state

## License

Academic use only. Dataset subject to IEEE DataPort terms of use.
