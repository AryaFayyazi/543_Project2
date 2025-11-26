#!/usr/bin/env python3
import csv
from collections import defaultdict
from math import isnan

import matplotlib.pyplot as plt

RESULTS_FILE = "results.csv"

def load_results(path=RESULTS_FILE):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # convert numeric fields
            for k in [
                "theta", "nkeys", "nqueries", "hot_threshold",
                "decay_alpha", "hot_fraction", "seed",
                "elapsed_sec", "qps",
                "hot_hits", "cold_hits", "not_found",
                "hot_keys", "cold_keys",
                "avg_hot_nodes_per_q", "avg_cold_nodes_per_q"
            ]:
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            rows.append(r)
    return rows

def group_key(row):
    """Group by workload + theta + nkeys + nqueries."""
    return (row["workload"], row["theta"], row["nkeys"], row["nqueries"])

def summarize(rows):
    grouped = defaultdict(dict)
    for r in rows:
        gk = group_key(r)
        grouped[gk][r["mode"]] = r
    return grouped

def print_comparison_table(grouped):
    print("\n=== Comparison summary (baseline vs hctree) ===")
    print("workload,theta,nkeys,nqueries,"
          "qps_baseline,qps_hctree,"
          "nodes_baseline,nodes_hctree,"
          "hot_keys_frac,hc_hot_hits_frac")
    for (workload, theta, nkeys, nqueries), modes in grouped.items():
        base = modes.get("baseline")
        hc = modes.get("hctree")
        if not base or not hc:
            continue
        qps_base = base["qps"]
        qps_hc = hc["qps"]
        nodes_base = base["avg_cold_nodes_per_q"]
        nodes_hc = hc["avg_hot_nodes_per_q"] + hc["avg_cold_nodes_per_q"]
        hot_frac = hc["hot_keys"] / hc["nkeys"] if hc["nkeys"] > 0 else 0.0
        hot_hits_frac = hc["hot_hits"] / hc["nqueries"] if hc["nqueries"] > 0 else 0.0

        print(f"{workload},{theta:.3f},{int(nkeys)},{int(nqueries)},"
              f"{qps_base:.1f},{qps_hc:.1f},"
              f"{nodes_base:.3f},{nodes_hc:.3f},"
              f"{hot_frac:.4f},{hot_hits_frac:.4f}")

def plot_qps_vs_mode(grouped):
    # One panel per (workload, theta)
    fig, ax = plt.subplots()

    labels = []
    base_vals = []
    hc_vals = []

    for (workload, theta, nkeys, nqueries), modes in grouped.items():
        base = modes.get("baseline")
        hc = modes.get("hctree")
        if not base or not hc:
            continue
        label = f"{workload}-θ={theta:.1f}"
        labels.append(label)
        base_vals.append(base["qps"])
        hc_vals.append(hc["qps"])

    x = range(len(labels))
    width = 0.35
    ax.bar([i - width/2 for i in x], base_vals, width, label="Baseline")
    ax.bar([i + width/2 for i in x], hc_vals, width, label="HCIndex")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Throughput (queries/sec)")
    ax.set_title("Throughput comparison: Baseline vs HCIndex")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_qps_vs_mode.png", dpi=300)
    plt.close(fig)

def plot_nodes_vs_mode(grouped):
    fig, ax = plt.subplots()

    labels = []
    base_nodes = []
    hc_nodes = []

    for (workload, theta, nkeys, nqueries), modes in grouped.items():
        base = modes.get("baseline")
        hc = modes.get("hctree")
        if not base or not hc:
            continue
        label = f"{workload}-θ={theta:.1f}"
        labels.append(label)
        base_nodes.append(base["avg_cold_nodes_per_q"])
        hc_nodes.append(hc["avg_hot_nodes_per_q"] + hc["avg_cold_nodes_per_q"])

    x = range(len(labels))
    width = 0.35
    ax.bar([i - width/2 for i in x], base_nodes, width, label="Baseline")
    ax.bar([i + width/2 for i in x], hc_nodes, width, label="HCIndex")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Avg. B-tree node visits / query")
    ax.set_title("Logical work per query: Baseline vs HCIndex")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_nodes_vs_mode.png", dpi=300)
    plt.close(fig)

def plot_hot_fraction_vs_theta(grouped):
    # For Zipf workloads only.
    thetas = []
    hot_frac = []
    hot_hits_frac = []

    for (workload, theta, nkeys, nqueries), modes in grouped.items():
        if workload != "zipf":
            continue
        hc = modes.get("hctree")
        if not hc:
            continue
        thetas.append(theta)
        hf = hc["hot_keys"] / hc["nkeys"] if hc["nkeys"] > 0 else 0.0
        hhf = hc["hot_hits"] / hc["nqueries"] if hc["nqueries"] > 0 else 0.0
        hot_frac.append(hf)
        hot_hits_frac.append(hhf)

    if not thetas:
        return

    # Sort by theta
    combined = sorted(zip(thetas, hot_frac, hot_hits_frac))
    thetas, hot_frac, hot_hits_frac = zip(*combined)

    fig, ax = plt.subplots()
    ax.plot(thetas, hot_frac, marker="o", label="Hot key fraction")
    ax.plot(thetas, hot_hits_frac, marker="s", label="Fraction of queries hitting hot")

    ax.set_xlabel("Zipf exponent θ")
    ax.set_ylabel("Fraction")
    ax.set_title("Hot tier utilization vs Zipf skew (HCIndex)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_hot_fraction_vs_theta.png", dpi=300)
    plt.close(fig)

def main():
    rows = load_results()
    grouped = summarize(rows)
    print_comparison_table(grouped)
    plot_qps_vs_mode(grouped)
    plot_nodes_vs_mode(grouped)
    plot_hot_fraction_vs_theta(grouped)
    print("\nGenerated: fig_qps_vs_mode.png, fig_nodes_vs_mode.png, fig_hot_fraction_vs_theta.png")

if __name__ == "__main__":
    main()
