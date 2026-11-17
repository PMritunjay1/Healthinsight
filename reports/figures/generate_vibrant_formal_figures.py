import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

out_dir = 'reports/figures'
os.makedirs(out_dir, exist_ok=True)

# ==============================================================================
# CLASSIC VIBRANT & HIGH-VISIBILITY COLOR PALETTE (Blue, Orange, Green, Purple)
# ==============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 13,
    'figure.titleweight': 'bold',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.9,
    'grid.color': '#E5E7EB',
    'grid.linestyle': '-',
    'grid.linewidth': 0.7,
    'grid.alpha': 0.7
})

# Classic vibrant, highly visible colors
C_BLUE   = '#1f77b4'   # Classic vibrant blue (HealthInsight / Cardio)
C_ORANGE = '#ff7f0e'   # Classic vibrant orange (Published reference)
C_GREEN  = '#2ca02c'   # Fresh emerald green (Pulmonology / Conformal)
C_PURPLE = '#9467bd'   # Royal purple (Neurology)
C_CYAN   = '#17becf'   # Cyan accent
C_SLATE  = '#7f7f7f'   # Neutral grey


# ==============================================================================
# 1. Figure 8.1: Single Unified Graph for Table 1 (All scores & comparisons on 1 graph)
# ==============================================================================
fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=300)
ax.set_axisbelow(True)

metrics = ['Disease Micro-F1 (%)', 'Disease Macro-F1 (%)', 'Disease Macro-AUROC (%)']
x = np.arange(len(metrics))
width = 0.19  # 4 bars per cluster

# Scores for each specialist and published reference
cardio_scores = [83.27, 63.02, 84.57]
pulmo_scores  = [69.14, 66.68, 87.79]
neuro_scores  = [71.58, 65.45, 89.86]
pub_scores    = [66.24, 59.29, 83.90]  # CAML for Micro/Macro F1, CardioMeta for AUROC

# Offsets for 4 bars centered around x using the exact same vibrant Blue, Orange, Green, Purple
r1 = ax.bar(x - 1.5*width, cardio_scores, width, label='Cardiologist V2 (HealthInsight)', 
            color=C_BLUE, edgecolor='#222222', linewidth=0.8, alpha=1.0, zorder=3)
r2 = ax.bar(x - 0.5*width, pulmo_scores, width, label='Pulmonologist V1 (HealthInsight)', 
            color=C_GREEN, edgecolor='#222222', linewidth=0.8, alpha=1.0, zorder=3)
r3 = ax.bar(x + 0.5*width, neuro_scores, width, label='Neurologist V1 (HealthInsight)', 
            color=C_PURPLE, edgecolor='#222222', linewidth=0.8, alpha=1.0, zorder=3)
r4 = ax.bar(x + 1.5*width, pub_scores, width, label='Published Benchmark (CAML / CardioMeta)', 
            color=C_ORANGE, edgecolor='#222222', linewidth=0.8, alpha=1.0, zorder=3)

# Values strictly in blank space above each bar (no text hiding bars)
for bar_group in [r1, r2, r3, r4]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 1.2, f'{h:.1f}%', 
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1A202C')

ax.set_ylabel('Performance Score (%)')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10.5, fontweight='bold')
ax.set_ylim([0, 116])  # Generous headroom so legend sits in completely blank whitespace
ax.set_title('Figure 8.1: Unified Disease Classification Performance vs. Published References\n(Single Graph Benchmarking Across All 3 Specialist Models, N = 39,594 Admissions)', pad=14)

# Legend placed in upper-center/left open blank whitespace
ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=9.2)
ax.grid(True, axis='y', zorder=0)

plt.tight_layout()
f1_path = os.path.join(out_dir, 'figure_table1_disease_classification_comparison.png')
plt.savefig(f1_path, dpi=300, bbox_inches='tight')
shutil.copy2(f1_path, os.path.join(out_dir, 'figure4_published_comparison.png'))
plt.close()
print(f"Saved: {f1_path}")


# ==============================================================================
# 2. Figure 8.3: Full-Cohort Specialty Accuracy (NO red 'I' error symbol, labels in blank space)
# DOTTED REFERENCE LINE CROSSES BARS ONLY IN THIS FIGURE (zorder=5)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=300)
ax.set_axisbelow(True)

cohort_labels = [
    'Full Cohort (Overall)\n(N = 39,594 | Route: 98.84%)',
    'Cardiology\n(N = 15,099 | Route: 99.93%)',
    'Pulmonology\n(N = 14,512 | Route: 96.97%)',
    'Neurology\n(N = 9,983 | Route: 99.89%)'
]
accs = [80.25, 91.32, 71.41, 76.35]

# Vibrant colors (Blue, Cyan, Green, Purple)
colors = [C_BLUE, '#1790cf', C_GREEN, C_PURPLE]

# Fully opaque bars (alpha=1.0, zorder=3) - NO yerr (NO red 'I' symbol!)
bars = ax.bar(cohort_labels, accs, color=colors, 
              alpha=1.0, edgecolor='#222222', linewidth=0.8, width=0.50, zorder=3)

# Accuracy values strictly on blank space above bars (no boxes inside bars hiding them!)
for bar, acc in zip(bars, accs):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.4, f'{acc:.2f}%', 
            ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#1A202C')

# Dotted reference baseline line crossing OVER the bars (zorder=5, linestyle=':')
ax.axhline(61.40, color='#D35400', linestyle=':', linewidth=2.4, 
           label='CAML Benchmark Reference (61.40% Micro-F1 [27])', zorder=5)

ax.set_ylabel('Top-1 Diagnostic Accuracy (%)')
ax.set_ylim([45, 108])  # Headroom ensures labels are in open blank space
ax.set_title('Figure 8.3: Full-Cohort Specialty Diagnostic Top-1 Accuracy\n(Audited Frozen Holdout Cohort, N = 39,594 Admissions)', pad=14)

# Legend placed in open upper-right blank whitespace (above Neurology/Pulmonology bars)
ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#CCCCCC')
ax.grid(True, axis='y', zorder=0)

plt.tight_layout()
f3_path = os.path.join(out_dir, 'figure_specialty_accuracy_confidence_intervals.png')
plt.savefig(f3_path, dpi=300)
plt.close()
print(f"Saved: {f3_path}")


# ==============================================================================
# 3. Figure 8.4: Per-Class Diagnostic F1-Score Across 18 Categories
# ==============================================================================
class_df = pd.read_csv('results/healthinsight_results_addendum_20261019_221000/per_class_metrics.csv')
c_df = class_df[~class_df['Disease Class'].isin(['MACRO AVERAGE', 'MICRO AVERAGE'])].copy()
c_df = c_df.sort_values(by='F1-Score (%)', ascending=True)

fig, ax = plt.subplots(figsize=(9.8, 7.4), dpi=300)
ax.set_axisbelow(True)

spec_color_map = {
    'coronary artery disease': C_BLUE,
    'heart failure': C_BLUE,
    'atrial fibrillation': C_BLUE,
    'hypertension': C_BLUE,
    'valvular heart disease': C_BLUE,
    'cardiogenic shock': C_BLUE,
    'chronic obstructive pulmonary disease': C_GREEN,
    'asthma': C_GREEN,
    'pneumonia': C_GREEN,
    'acute respiratory distress syndrome': C_GREEN,
    'pleural effusion': C_GREEN,
    'pulmonary embolism': C_GREEN,
    'stroke': C_PURPLE,
    'epilepsy': C_PURPLE,
    'altered mental status': C_PURPLE,
    'intracranial hemorrhage': C_PURPLE,
    'encephalopathy': C_PURPLE,
    'migraine': C_PURPLE
}

bar_colors = [spec_color_map.get(c.lower(), C_SLATE) for c in c_df['Disease Class']]

# Fully opaque horizontal bars (alpha=1.0, zorder=3)
bars = ax.barh(c_df['Disease Class'].str.title(), c_df['F1-Score (%)'], color=bar_colors, 
               edgecolor='#222222', linewidth=0.8, alpha=1.0, height=0.62, zorder=3)

# Values strictly in blank space to the right of bars
for bar, f1, sup in zip(bars, c_df['F1-Score (%)'], c_df['Ground Truth Count in Cohort']):
    xval = bar.get_width()
    ax.text(xval + 1.0, bar.get_y() + bar.get_height()/2.0, f'{f1:.1f}% (N={sup:,})', 
            ha='left', va='center', fontsize=8.5, fontweight='bold', color='#1A202C')

# Contextual reference lines in distinct vibrant colors
ax.axvline(54.55, color='#e67e22', linestyle='--', linewidth=1.5, label='Macro-Average F1: 54.55%', zorder=2)
ax.axvline(64.71, color=C_BLUE, linestyle='-.', linewidth=1.5, label='Micro-Average F1: 64.71%', zorder=2)
ax.axvline(53.20, color=C_ORANGE, linestyle=':', linewidth=1.5, label='CAML Macro-F1 Reference (53.20% [27])', zorder=2)

ax.set_xlabel('F1-Score (%)')
ax.set_xlim([0, 100])
ax.set_title('Figure 8.4: Per-Class Diagnostic F1-Score Across 18 Specialized Disease Categories\n(Cardiology: Blue, Pulmonology: Green, Neurology: Purple; N = 39,594)', pad=14)

# Legend in blank whitespace at bottom right
ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#CCCCCC')
ax.grid(True, axis='x', zorder=0)

plt.tight_layout()
f4_path = os.path.join(out_dir, 'figure_per_class_f1_distribution.png')
plt.savefig(f4_path, dpi=300)
plt.close()
print(f"Saved: {f4_path}")


# ==============================================================================
# 4. Figure 8.5: Calibration Reliability Diagram (Table 10)
# ==============================================================================
src_cal = 'results/calibration_reliability_diagram.png'
f5_path = os.path.join(out_dir, 'figure_reliability_diagram_heuristic.png')
shutil.copy2(src_cal, f5_path)
print(f"Saved clean reliability diagram to: {f5_path}")


# ==============================================================================
# 5. Figure 8.6: Split-Conformal Selective Accuracy vs. Coverage Trade-Off (Table 11)
# ==============================================================================
rc_df = pd.read_csv('results/healthinsight_results_addendum_20261019_221000/risk_coverage_curve.csv')

fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=300)
ax.set_axisbelow(True)

cov_pct = rc_df['Coverage (%)'].values
sel_acc = rc_df['Selective Accuracy (%)'].values

# Vibrant blue curve with clear circular markers
ax.plot(cov_pct, sel_acc, 'o-', color=C_BLUE, linewidth=2.4, markersize=7, 
        label='HealthInsight-UQ Conformal Frontier', zorder=3)

# Contextual reference lines
ax.axhline(80.25, color=C_SLATE, linestyle='--', linewidth=1.5, 
           label='Unconditioned Top-1 Accuracy (80.25%, Coverage: 100%)', zorder=2)
ax.axhline(82.50, color=C_ORANGE, linestyle=':', linewidth=1.8, 
           label='Flan-PaLM 540B Contextual Reference (82.50% at 55% Cov [19])', zorder=2)

ax.set_xlabel('Empirical System Coverage (%)')
ax.set_ylabel('Selective Top-1 Diagnostic Accuracy (%)')
ax.set_xlim([30, 105])
ax.set_ylim([79.0, 83.5])
ax.set_title('Figure 8.6: Split-Conformal Selective Accuracy vs. Coverage Trade-Off\n(Audited Hold-Out Cohort, N = 39,594 Admissions)', pad=14)

# Legend strictly in open lower-left blank whitespace (no text over lines)
ax.legend(loc='lower left', frameon=True, framealpha=0.95, edgecolor='#CCCCCC')
ax.grid(True, zorder=0)

plt.tight_layout()
f6_path = os.path.join(out_dir, 'figure_coverage_vs_selective_accuracy.png')
plt.savefig(f6_path, dpi=300)
plt.close()
print(f"Saved: {f6_path}")


# ==============================================================================
# 6. Figure 8.7: Consensus Ablations & Comparative Literature Benchmarks (Table 12)
# ==============================================================================
fig, ax = plt.subplots(figsize=(9.8, 5.4), dpi=300)
ax.set_axisbelow(True)

models = [
    'CAML Baseline [27]',
    'MedAide Baseline [4]',
    'Uniform Majority Voting',
    'Single Specialist Alone',
    'Legacy RAC Heuristic',
    'Calibrated Beta Trust (Ours)',
    'Conformal Selective (tau=0.55)'
]
scores = [61.40, 71.00, 78.12, 79.80, 79.94, 80.25, 82.24]

# Vibrant colors (Orange for Baselines, Slate for internal ablations, Blue for Full, Green for Conformal)
palette = [C_ORANGE, '#f39c12', '#95a5a6', C_SLATE, '#3498db', C_BLUE, C_GREEN]

# Fully opaque bars (alpha=1.0, zorder=3)
bars = ax.bar(models, scores, color=palette, edgecolor='#222222', width=0.50, alpha=1.0, linewidth=0.8, zorder=3)

# Values strictly in blank space above bars
for bar, s in zip(bars, scores):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{s:.2f}%', 
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1A202C')

ax.set_ylabel('Diagnostic Accuracy / Micro-F1 (%)')
ax.set_ylim([50, 92])
ax.set_title('Figure 8.7: Comparative Multi-Agent Consensus Ablation and Published Literature Benchmarks\n(Evaluated on Hold-out Clinical Cohort, N = 39,594)', pad=14)

# 22-degree rotation with right-alignment to completely prevent label overlap
plt.xticks(rotation=22, ha='right', fontsize=9, fontweight='medium')

# Clean legend in upper-left blank whitespace
custom_lines = [
    plt.Rectangle((0, 0), 1, 1, facecolor=C_ORANGE, edgecolor='#222222'),
    plt.Rectangle((0, 0), 1, 1, facecolor=C_SLATE, edgecolor='#222222'),
    plt.Rectangle((0, 0), 1, 1, facecolor=C_BLUE, edgecolor='#222222'),
    plt.Rectangle((0, 0), 1, 1, facecolor=C_GREEN, edgecolor='#222222')
]
ax.legend(custom_lines, ['Published Literature Baselines [4], [27]', 'Multi-Agent Internal Ablations', 'HealthInsight-UQ Full Cohort', 'HealthInsight-UQ Conformal Gated'], 
          loc='upper left', frameon=True, framealpha=0.95, edgecolor='#CCCCCC')
ax.grid(True, axis='y', zorder=0)

plt.tight_layout()
f7_path = os.path.join(out_dir, 'figure_consensus_ablation_comparison.png')
plt.savefig(f7_path, dpi=300)
plt.close()
print(f"Saved: {f7_path}")


# ==============================================================================
# 7. Figure 8.8: Computational Latency and Memory Footprint (Table 14)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=300)

# Subplot (a): Latency
ax1.set_axisbelow(True)
engines = ['FP32 PyTorch\n(CPU Base)', 'INT8 ONNX\n(CPU Opt)', 'INT8 ONNX\n(RTX 4050 GPU)']
latencies = [2521.0, 1574.0, 411.5]
lat_colors = [C_SLATE, C_BLUE, C_GREEN]

bars1 = ax1.bar(engines, latencies, color=lat_colors, edgecolor='#222222', width=0.48, alpha=1.0, linewidth=0.8, zorder=3)
for b, lat in zip(bars1, latencies):
    y = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2.0, y + 60, f'{lat:.1f} ms\n({2521.0/lat:.1f}x)', 
             ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1A202C')

ax1.set_ylabel('Mean Execution Latency per Encounter (ms)')
ax1.set_ylim([0, 3200])
ax1.set_title('(a) Runtime Latency Across Engines', pad=10)
ax1.grid(True, axis='y', zorder=0)

# Subplot (b): Memory
ax2.set_axisbelow(True)
mem_configs = ['Baseline PyTorch\nSystem RAM', 'HealthInsight-UQ\nSystem RAM', 'Laptop Budget\nUpper Limit', 'Active GPU\nVRAM Usage']
mem_vals = [11.2, 4.1, 16.0, 2.8]
mem_colors = [C_ORANGE, C_BLUE, C_SLATE, C_PURPLE]

bars2 = ax2.bar(mem_configs, mem_vals, color=mem_colors, edgecolor='#222222', width=0.50, alpha=1.0, linewidth=0.8, zorder=3)
for b, mem in zip(bars2, mem_vals):
    y = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2.0, y + 0.35, f'{mem:.1f} GB', 
             ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1A202C')

ax2.axhline(16.0, color=C_ORANGE, linestyle='--', linewidth=1.6, label='16 GB Workstation Budget Constraint', zorder=4)
ax2.set_ylabel('Resident Memory Footprint (GB)')
ax2.set_ylim([0, 20.0])
ax2.set_title('(b) Peak Hardware Memory Footprint', pad=10)
ax2.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#CCCCCC')
ax2.grid(True, axis='y', zorder=0)

plt.suptitle('Figure 8.8: Computational Efficiency, Acceleration Speedup, and Laptop-Class Memory Compliance', fontsize=12.5, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])

f8_path = os.path.join(out_dir, 'figure_latency_and_hardware_profile.png')
plt.savefig(f8_path, dpi=300)
plt.close()
print(f"Saved: {f8_path}")

print("\nALL REFINED VIBRANT FIGURES GENERATED CLEANLY WITH 100% OPAQUE BARS AND LABELS IN BLANK SPACE!")
