"""
Problem 4.5: does lexical diversity (Yule's K, Herdan's C, Guiraud's R, TTR) separate
templated/near-duplicate review clusters (a fraud-like proxy, since no ground-truth
fraud labels exist in this dataset) from organic reviews, on Amazon app reviews?

Stages:
  1. Load + clean amazon.csv
  2. Near-duplicate clustering via MinHash LSH (proxy for "templated/fraud-like")
  3. Per-review lexical diversity metrics
  4. Statistical comparison: templated vs organic
  5. LNRE G/Q tail-contribution comparison between the two groups
  6. Save charts + summary table
"""
import re
import json
import string
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datasketch import MinHash, MinHashLSH

OUT_DIR = "E:/Training/project_pub/fraud_lexdiv_amazon"

# ---------------------------------------------------------------- Stage 1
print("Stage 1: load + clean")

df = pd.read_csv("E:/Training/project_pub/amazon.csv")
df["content"] = df["content"].fillna("").astype(str)


def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean"] = df["content"].apply(clean)
df["tokens"] = df["clean"].apply(str.split)
df["n_tokens"] = df["tokens"].apply(len)

# need enough tokens per review for diversity metrics to be meaningful
df = df[df["n_tokens"] >= 8].reset_index(drop=True)
print(f"Reviews retained (>=8 tokens): {len(df)}")

# ---------------------------------------------------------------- Stage 2
print("Stage 2: near-duplicate clustering via MinHash LSH")


def shingles(tokens, k=3):
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def minhash_for(tokens, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for sh in shingles(tokens):
        m.update(sh.encode("utf8"))
    return m


DUP_THRESHOLD = 0.70
lsh = MinHashLSH(threshold=DUP_THRESHOLD, num_perm=128)
minhashes = {}
for idx, tokens in zip(df.index, df["tokens"]):
    m = minhash_for(tokens)
    minhashes[idx] = m
    lsh.insert(str(idx), m)

# union-find to group near-duplicates into clusters
parent = {idx: idx for idx in df.index}


def find(x):
    while parent[x] != x:
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


for idx in df.index:
    neighbors = lsh.query(minhashes[idx])
    for n in neighbors:
        n = int(n)
        if n != idx:
            union(idx, n)

cluster_id = {idx: find(idx) for idx in df.index}
df["cluster"] = df.index.map(cluster_id)
cluster_sizes = df["cluster"].value_counts()
df["cluster_size"] = df["cluster"].map(cluster_sizes)

templated_mask = df["cluster_size"] >= 3  # appears with >=2 near-duplicates
df["group"] = np.where(templated_mask, "templated", "organic")

print(df["group"].value_counts())
n_clusters_templated = df.loc[templated_mask, "cluster"].nunique()
print(f"Templated clusters (size>=3): {n_clusters_templated}")

# ---------------------------------------------------------------- Stage 3
print("Stage 3: per-review lexical diversity metrics")


def yules_k(tokens):
    n = len(tokens)
    if n < 2:
        return np.nan
    freqs = Counter(tokens)
    m1 = n
    m2 = sum(f * f for f in freqs.values())
    return 1e4 * (m2 - m1) / (m1 * m1)


def herdans_c(tokens):
    n = len(tokens)
    v = len(set(tokens))
    if n <= 1 or v <= 0:
        return np.nan
    return np.log(v) / np.log(n)


def guirauds_r(tokens):
    n = len(tokens)
    v = len(set(tokens))
    if n <= 0:
        return np.nan
    return v / np.sqrt(n)


def ttr(tokens):
    n = len(tokens)
    v = len(set(tokens))
    if n <= 0:
        return np.nan
    return v / n


df["yules_k"] = df["tokens"].apply(yules_k)
df["herdans_c"] = df["tokens"].apply(herdans_c)
df["guirauds_r"] = df["tokens"].apply(guirauds_r)
df["ttr"] = df["tokens"].apply(ttr)

# ---------------------------------------------------------------- Stage 4
print("Stage 4: statistical comparison (templated vs organic)")

metrics = ["yules_k", "herdans_c", "guirauds_r", "ttr"]
summary_rows = []
for m in metrics:
    org = df.loc[df["group"] == "organic", m].dropna()
    tmp = df.loc[df["group"] == "templated", m].dropna()
    u_stat, p_val = stats.mannwhitneyu(tmp, org, alternative="two-sided")
    # rank-biserial effect size
    n1, n2 = len(tmp), len(org)
    effect = 1 - (2 * u_stat) / (n1 * n2)
    summary_rows.append({
        "metric": m,
        "organic_median": org.median(),
        "templated_median": tmp.median(),
        "p_value": p_val,
        "rank_biserial_effect": effect,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df["organic_n"] = (df["group"] == "organic").sum()
summary_df["templated_n"] = (df["group"] == "templated").sum()
print(summary_df.to_string(index=False))
summary_df.to_csv(f"{OUT_DIR}/lexdiv_summary_t{DUP_THRESHOLD}.csv", index=False)

# ---------------------------------------------------------------- Stage 5
print("Stage 5: LNRE G/Q tail-contribution comparison")


def q_g_tail_contribution(token_lists):
    all_tokens = [t for toks in token_lists for t in toks]
    freqs = Counter(all_tokens)
    sorted_freqs = sorted(freqs.values(), reverse=True)
    total = sum(sorted_freqs)
    cum = np.cumsum(sorted_freqs) / total
    # tail = bottom 90% of ranks (by count of distinct tokens), contribution to total mass
    tail_rank_idx = int(len(sorted_freqs) * 0.9)
    tail_contribution = 1 - cum[tail_rank_idx] if tail_rank_idx < len(cum) else 0.0
    return cum, tail_contribution, len(sorted_freqs), total


cum_org, tail_org, vocab_org, total_org = q_g_tail_contribution(df.loc[df["group"] == "organic", "tokens"])
cum_tmp, tail_tmp, vocab_tmp, total_tmp = q_g_tail_contribution(df.loc[df["group"] == "templated", "tokens"])

print(f"Organic:   vocab={vocab_org}, tokens={total_org}, tail(bottom90%-rank) contribution={tail_org:.4f}")
print(f"Templated: vocab={vocab_tmp}, tokens={total_tmp}, tail(bottom90%-rank) contribution={tail_tmp:.4f}")

with open(f"{OUT_DIR}/qg_tail_summary_t{DUP_THRESHOLD}.json", "w") as f:
    json.dump({
        "organic": {"vocab": vocab_org, "tokens": int(total_org), "tail_contribution": tail_org},
        "templated": {"vocab": vocab_tmp, "tokens": int(total_tmp), "tail_contribution": tail_tmp},
    }, f, indent=2)

# ---------------------------------------------------------------- Stage 6
print("Stage 6: charts")

plt.figure(figsize=(8, 5))
plt.plot(np.linspace(0, 1, len(cum_org)), cum_org, label="Organic")
plt.plot(np.linspace(0, 1, len(cum_tmp)), cum_tmp, label="Templated")
plt.xlabel("Rank percentile")
plt.ylabel("Cumulative token-mass contribution")
plt.title("Cumulative contribution: organic vs. templated reviews")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cumulative_contribution_t{DUP_THRESHOLD}.png", dpi=200)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, m in zip(axes.ravel(), metrics):
    data = [df.loc[df["group"] == "organic", m].dropna(), df.loc[df["group"] == "templated", m].dropna()]
    ax.boxplot(data, tick_labels=["Organic", "Templated"], showfliers=False)
    ax.set_title(m)
plt.suptitle("Lexical diversity metrics: organic vs. templated review clusters")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/lexdiv_boxplots_t{DUP_THRESHOLD}.png", dpi=200)
plt.close()

df.drop(columns=["tokens"]).to_csv(f"{OUT_DIR}/reviews_with_groups_t{DUP_THRESHOLD}.csv", index=False)

print("Done. Outputs in", OUT_DIR)
