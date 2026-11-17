import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_reliability_diagram():
    # Load calibration features
    calib_path = os.path.join('datasets', 'processed', 'calib_features.csv')
    if not os.path.exists(calib_path):
        print("calib_features.csv not found")
        return
        
    df = pd.read_csv(calib_path)
    confs = df['calibrated_confidence'].values
    accs = df['correctness'].values
    
    num_bins = 10
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    
    bin_accs = []
    bin_confs = []
    
    for i in range(num_bins):
        if i == num_bins - 1:
            bin_mask = (confs >= bins[i]) & (confs <= bins[i+1])
        else:
            bin_mask = (confs >= bins[i]) & (confs < bins[i+1])
            
        if np.sum(bin_mask) > 0:
            bin_accs.append(np.mean(accs[bin_mask]))
            bin_confs.append(np.mean(confs[bin_mask]))
            
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    plt.plot(bin_confs, bin_accs, 's-', label="Logistic Calibrated")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    
    out_dir = os.path.join('results', 'graphs')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'reliability_diagram.png'))
    print("Reliability diagram saved to results/graphs/reliability_diagram.png")

if __name__ == "__main__":
    generate_reliability_diagram()
