"""
scalability_analysis.py
=======================
Benchmarks the three trained models across growing batch sizes and
prints a summary table.  Run from the project root:

    python scalability_analysis.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
APP_DIR = os.path.join(os.path.dirname(__file__), "app")

def load_artifacts():
    models = {
        "Ridge":              joblib.load(os.path.join(APP_DIR, "ridge_model.pkl")),
        "Random Forest":      joblib.load(os.path.join(APP_DIR, "rf_model.pkl")),
        "Gradient Boosting":  joblib.load(os.path.join(APP_DIR, "gb_model.pkl")),
    }
    preprocessor = joblib.load(os.path.join(APP_DIR, "preprocessor.pkl"))
    df = pd.read_csv(os.path.join(APP_DIR, "github_features_clean.csv"))
    return models, preprocessor, df

def benchmark(models, preprocessor, df, batch_sizes, n_repeats=20):
    features = [
        "topic_count", "forks_count_log", "open_issues_count_log",
        "days_since_last_push_log", "has_projects", "has_wiki",
    ]
    X_full = preprocessor.transform(df[features])

    results = []
    for bs in batch_sizes:
        X_batch = X_full[:bs]
        row = {"batch_size": bs}
        for name, model in models.items():
            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                model.predict(X_batch)
                times.append((time.perf_counter() - t0) * 1000)
            row[name] = round(np.mean(times), 3)
        results.append(row)
    return pd.DataFrame(results)

def throughput(models, preprocessor, df, n_repeats=100):
    features = [
        "topic_count", "forks_count_log", "open_issues_count_log",
        "days_since_last_push_log", "has_projects", "has_wiki",
    ]
    X_full = preprocessor.transform(df[features])
    n = len(X_full)

    print("\n── Throughput (repos / second) ─────────────────────────────")
    for name, model in models.items():
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            model.predict(X_full)
        elapsed = time.perf_counter() - t0
        rps = (n_repeats * n) / elapsed
        print(f"  {name:<22}  {rps:>12,.0f} repos/sec")

def model_sizes():
    print("\n── Model File Sizes ────────────────────────────────────────")
    for label, fname in [
        ("Ridge",             "ridge_model.pkl"),
        ("Random Forest",     "rf_model.pkl"),
        ("Gradient Boosting", "gb_model.pkl"),
    ]:
        path = os.path.join(APP_DIR, fname)
        kb = os.path.getsize(path) / 1024
        print(f"  {label:<22}  {kb:>8.1f} KB")

def main():
    print("Loading models …")
    models, preprocessor, df = load_artifacts()

    batch_sizes = [1, 10, 50, 100, 250, 500, 1000]

    print("\n── Inference Latency (ms, mean over 20 runs) ───────────────")
    df_lat = benchmark(models, preprocessor, df, batch_sizes)
    print(df_lat.to_string(index=False))

    throughput(models, preprocessor, df)
    model_sizes()

    print("\nDone.")

if __name__ == "__main__":
    main()

