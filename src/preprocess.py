"""
preprocess.py — Data Loading and Preprocessing
================================================
Loads the raw tweet dataset, cleans text, extracts keywords,
and splits into depressive / non-depressive subsets.

Usage:
    python -m src.preprocess
"""

import os
import re
import pandas as pd
from collections import Counter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DATA_PATH = os.path.join("data", "raw", "clean_tweet_Dec19ToDec20.csv")
PROCESSED_DIR = os.path.join("data", "processed")

# English stopwords — a curated list covering common function words.
# We keep the list self-contained so there is no dependency on NLTK.
STOPWORDS = frozenset([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    # Additional common Twitter / conversational words to filter out
    "like", "get", "got", "go", "going", "would", "could", "also",
    "one", "two", "even", "still", "much", "make", "know", "think",
    "want", "say", "said", "see", "come", "take", "let", "really",
    "back", "right", "well", "good", "new", "first", "last", "time",
    "way", "thing", "things", "day", "people", "need", "many", "may",
    "us", "amp",  # &amp; remnant common in tweet data
])


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and perform initial validation."""
    print(f"[preprocess] Loading raw data from {path} ...")
    df = pd.read_csv(path)

    # Drop the unnamed index column carried over from the source
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Validate expected columns
    assert "text" in df.columns, "Missing 'text' column"
    assert "sentiment" in df.columns, "Missing 'sentiment' column"

    print(f"[preprocess] Raw dataset: {len(df)} rows")
    return df


def clean_text(text: str) -> str:
    """
    Clean a single tweet text string.
    - Remove residual URLs
    - Remove non-alphabetic characters (keep spaces)
    - Collapse whitespace
    - Lowercase (should already be, but enforce)
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove non-alpha characters (keep spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Collapse whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def extract_keywords(text: str) -> list:
    """
    Tokenize text and remove stopwords.
    Returns a list of meaningful keywords (length >= 3).
    """
    if not text:
        return []
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 3]


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning and keyword extraction to the full dataset.

    Adds columns:
    - clean_text: cleaned tweet text
    - keywords: list of keywords per tweet
    - keyword_count: number of keywords per tweet
    """
    print("[preprocess] Cleaning text ...")
    # Drop rows with null text
    df = df.dropna(subset=["text"]).copy()
    df["clean_text"] = df["text"].apply(clean_text)

    # Remove rows that became empty after cleaning
    df = df[df["clean_text"].str.len() > 0].copy()

    print("[preprocess] Extracting keywords ...")
    df["keywords"] = df["clean_text"].apply(extract_keywords)
    df["keyword_count"] = df["keywords"].apply(len)

    # Remove rows with no meaningful keywords
    df = df[df["keyword_count"] > 0].copy()

    print(f"[preprocess] After cleaning: {len(df)} tweets")
    return df


def split_by_sentiment(df: pd.DataFrame) -> tuple:
    """Split dataset into depressive (1) and non-depressive (0) subsets."""
    dep = df[df["sentiment"] == 1].copy()
    nondep = df[df["sentiment"] == 0].copy()
    print(f"[preprocess] Depressive tweets:     {len(dep)}")
    print(f"[preprocess] Non-depressive tweets:  {len(nondep)}")
    return dep, nondep


def get_top_keywords(df: pd.DataFrame, top_n: int = 200) -> list:
    """
    Get the top N most frequent keywords from a subset.
    Returns a list of keyword strings, ordered by frequency.
    """
    counter = Counter()
    for kw_list in df["keywords"]:
        counter.update(kw_list)
    return [word for word, _ in counter.most_common(top_n)]


def save_processed_data(df: pd.DataFrame, dep: pd.DataFrame,
                        nondep: pd.DataFrame) -> None:
    """Save processed dataframes to CSV for downstream use."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save full processed dataset (keywords stored as pipe-separated)
    df_save = df[["clean_text", "sentiment", "keyword_count"]].copy()
    df_save["keywords"] = df["keywords"].apply(lambda x: "|".join(x))
    df_save.to_csv(os.path.join(PROCESSED_DIR, "all_tweets_processed.csv"),
                   index=False)

    # Save top keywords per subset
    top_dep = get_top_keywords(dep, 200)
    top_nondep = get_top_keywords(nondep, 200)

    pd.DataFrame({"keyword": top_dep}).to_csv(
        os.path.join(PROCESSED_DIR, "top_keywords_depressive.csv"), index=False)
    pd.DataFrame({"keyword": top_nondep}).to_csv(
        os.path.join(PROCESSED_DIR, "top_keywords_nondepressive.csv"), index=False)

    print(f"[preprocess] Saved processed data to {PROCESSED_DIR}/")


def print_summary(dep: pd.DataFrame, nondep: pd.DataFrame) -> None:
    """Print a quick summary of the preprocessed data."""
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    dep_kw = Counter()
    for kw_list in dep["keywords"]:
        dep_kw.update(kw_list)

    nondep_kw = Counter()
    for kw_list in nondep["keywords"]:
        nondep_kw.update(kw_list)

    print(f"\nDepressive tweets:      {len(dep)}")
    print(f"  Unique keywords:      {len(dep_kw)}")
    print(f"  Avg keywords/tweet:   {dep['keyword_count'].mean():.1f}")
    print(f"  Top 10 keywords:      {[w for w, _ in dep_kw.most_common(10)]}")

    print(f"\nNon-depressive tweets:  {len(nondep)}")
    print(f"  Unique keywords:      {len(nondep_kw)}")
    print(f"  Avg keywords/tweet:   {nondep['keyword_count'].mean():.1f}")
    print(f"  Top 10 keywords:      {[w for w, _ in nondep_kw.most_common(10)]}")

    # Overlap analysis
    dep_set = set(dep_kw.keys())
    nondep_set = set(nondep_kw.keys())
    overlap = dep_set & nondep_set
    dep_only = dep_set - nondep_set
    nondep_only = nondep_set - dep_set

    print(f"\nKeyword overlap:        {len(overlap)} shared keywords")
    print(f"  Depressive-only:      {len(dep_only)} unique keywords")
    print(f"  Non-depressive-only:  {len(nondep_only)} unique keywords")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run the full preprocessing pipeline."""
    # Load
    df = load_raw_data()

    # Clean & extract
    df = preprocess_dataset(df)

    # Split
    dep, nondep = split_by_sentiment(df)

    # Save
    save_processed_data(df, dep, nondep)

    # Summary
    print_summary(dep, nondep)

    print("\n[preprocess] Done. Run `python -m src.build_graph` next.")


if __name__ == "__main__":
    main()
