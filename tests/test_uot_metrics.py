import torch
import numpy as np
import ot
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests')))
from utils import make_dying_example_unif
from mioflow import TimeSeriesDataset

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
    
    # Check distances between contiguous time points
    print("--- Squared Distances (Cost Matrix M) ---")
    dists = []
    for i in range(len(dataset)):
        batch = dataset[i]
        x0 = batch['X_start'].cpu().numpy()
        x1 = batch['X_end'].cpu().numpy()
        
        M = ot.dist(x0, x1) # Squared Euclidean
        
        # What is the distance if we just matched everyone to their closest neighbor?
        min_dists = M.min(axis=1)
        mean_min_dist = min_dists.mean()
        max_min_dist = min_dists.max()
        
        print(f"Time {i}->{i+1}: Mean nearest neighbor squared distance: {mean_min_dist:.4f}, Max: {max_min_dist:.4f}")
        dists.extend(min_dists.tolist())
        
    print(f"\nOverall minimum distance to survive: {min(dists):.4f}")
    print(f"Overall average distance to survive: {np.mean(dists):.4f}")
    print(f"Overall maximum distance to survive (for a dying point with no target!): {max(dists):.4f}")

if __name__ == "__main__":
    main()
