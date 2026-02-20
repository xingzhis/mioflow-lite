import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sklearn.preprocessing import StandardScaler
from utils import make_dying_example_unif
from mioflow import TimeSeriesDataset, compute_uot_growth_rates

def main():
    df = make_dying_example_unif(n_pts_per_bin=50, seed=223)
    time_pts = sorted(df['samples'].unique())
    time_series_data = []
    
    for t in time_pts:
        pts = df[df['samples'] == t][['d1', 'd2']].values
        time_series_data.append((pts, float(t)))
        
    all_pts = np.vstack([x for x, t in time_series_data])
    scaler = StandardScaler()
    scaler.fit(all_pts)
    
    scaled_data = []
    for x, t in time_series_data:
        scaled_data.append((scaler.transform(x), t))
        
    dataset = TimeSeriesDataset(scaled_data)
    
    for div in ['l2', 'kl']:
        # Try various reg_m values to see what produces crisp values near 1.0 (good) and 0.0 (bad)
        for reg_m in [0.5, 1.0, 2.0, 5.0, 10.0]:
            try:
                uos = compute_uot_growth_rates(dataset, reg_m=[reg_m, 100.0], div=div)
                all_mass = torch.cat(uos).numpy()
                mean = all_mass.mean()
                max_val = all_mass.max()
                min_val = all_mass.min()
                print(f"div={div}, reg_m={reg_m:<4.1f} | Mean: {mean:.3f}, Max: {max_val:.3f}, Min: {min_val:.3f}")
            except Exception as e:
                pass


if __name__ == "__main__":
    main()
