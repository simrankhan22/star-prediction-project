import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

APP_DIR = "app"  # Adjust if running from elsewhere

# 1. Load data and models
print("Loading models and preprocessors...")
ridge        = joblib.load(f"{APP_DIR}/ridge_model.pkl")
rf           = joblib.load(f"{APP_DIR}/rf_model.pkl")
gb           = joblib.load(f"{APP_DIR}/gb_model.pkl")
preprocessor = joblib.load(f"{APP_DIR}/preprocessor.pkl")

df = pd.read_csv(f"{APP_DIR}/github_features_clean.csv")
features = ['topic_count', 'forks_count_log', 'open_issues_count_log',
            'days_since_last_push_log', 'has_projects', 'has_wiki']
X_scaled = preprocessor.transform(df[features])

models = {"Ridge": ridge, "Gradient Boosting": gb, "Random Forest": rf}
batch_sizes = [1, 10, 50, 100, 250, 500, 1000]

# Dictionaries to store metrics dynamically for plotting
latency_results = {name: [] for name in models.keys()}
throughput_results = {}

# 2. Latency Benchmark Execution
print("\n=== RUNNING INFERENCE LATENCY BENCHMARK ===")
print(f"{'Batch':>6} | {'Ridge':>10} | {'Grad. Boost':>12} | {'Rand. Forest':>13}")
print("-" * 55)

for bs in batch_sizes:
    row = f"{bs:>6} |"
    for name, model in models.items():
        times = []
        # Run 20 iterations to get a reliable average
        for _ in range(20):
            t0 = time.perf_counter()
            model.predict(X_scaled[:bs])
            times.append((time.perf_counter() - t0) * 1000)  # convert to ms
        
        avg_latency = np.mean(times)
        latency_results[name].append(avg_latency)
        row += f" {avg_latency:>10.3f} ms |"
    print(row)

# 3. Throughput Benchmark Execution
print("\n=== RUNNING THROUGHPUT BENCHMARK ===")
for name, model in models.items():
    t0 = time.perf_counter()
    for _ in range(100):
        model.predict(X_scaled)
    elapsed = time.perf_counter() - t0
    rps = (100 * len(X_scaled)) / elapsed
    throughput_results[name] = rps
    print(f"  {name:<22}: {rps:>12,.0f} repos/sec")

# 4. Generate and Save Plot
print("\nGenerating performance visualization graphs...")

# Set professional font styles
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Initialize a 1x2 subplot panel (without using .figure())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Plot 1: Latency Curve (Left Subplot) ---
colors = {"Ridge": "#1f77b4", "Gradient Boosting": "#ff7f0e", "Random Forest": "#2ca02c"}
markers = {"Ridge": "o", "Gradient Boosting": "s", "Random Forest": "^"}

for name in models.keys():
    ax1.plot(batch_sizes, latency_results[name], 
             marker=markers[name], linewidth=2, 
             label=name, color=colors[name])

ax1.set_title('Inference Latency vs. Batch Size', fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Batch Size (Number of Repositories)', fontsize=11, labelpad=10)
ax1.set_ylabel('Latency (milliseconds)', fontsize=11, labelpad=10)
ax1.set_xscale('log')
ax1.set_xticks(batch_sizes)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=10, loc='upper left')

# --- Plot 2: Throughput Bar Chart (Right Subplot) ---
# Sort the throughput items from lowest to highest for better presentation
sorted_throughput = sorted(throughput_results.items(), key=lambda x: x[1])
sorted_models = [item[0] for item in sorted_throughput]
sorted_values = [item[1] for item in sorted_throughput]
bar_colors = [colors[model] for model in sorted_models]

bars = ax2.bar(sorted_models, sorted_values, color=bar_colors, width=0.5)
ax2.set_title('Model Throughput Comparison', fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel('Model Type', fontsize=11, labelpad=10)
ax2.set_ylabel('Throughput (Repositories / Second)', fontsize=11, labelpad=10)
ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add precise text labels on top of the bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (max(sorted_values) * 0.02),
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Automatically handle margins and prevent label overlaps
plt.tight_layout()

# Save image file to the workspace
output_image = 'scalability_analysis.png'
plt.savefig(output_image, dpi=300)
print(f"Success! Scalability chart successfully saved as '{output_image}'")