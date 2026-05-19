import numpy as np
import pandas as pd

def make_uniform_rect(x_bounds, y_bounds, num_points):
    """
    Generates uniformly distributed points within a rectangular box.

    Parameters:
    - x_bounds: Tuple (x_min, x_max) defining the bounds on the X-axis.
    - y_bounds: Tuple (y_min, y_max) defining the bounds on the Y-axis.
    - num_points: The number of points to generate.

    Returns:
    - points: A 2D numpy array where each row is a point [x, y].
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    
    x_points = np.random.uniform(x_min, x_max, num_points)
    y_points = np.random.uniform(y_min, y_max, num_points)

    points = np.column_stack((x_points, y_points))
    
    return points

def make_dying_example_unif(n_pts_per_bin=50, seed=223):
    np.random.seed(seed)
    tp0 = make_uniform_rect((-1.5, -0.4), (-0.5, 0.5), n_pts_per_bin*2)
    tp1 = make_uniform_rect((-0.6, 0.6), (-0.5, 0.5), n_pts_per_bin*2)
    tp2 = make_uniform_rect((0.4, 1.6), (-0.5, 0.5), n_pts_per_bin*2)
    tp3 = make_uniform_rect((1.4, 2.6), (0., 0.5), n_pts_per_bin)
    tp4 = make_uniform_rect((2.4, 3.1), (0., 0.5), n_pts_per_bin)
    # concatenate, but add a column for the timepoint, make it a dataframe
    # Create dataframes and add a timepoint column

    df_tp0 = pd.DataFrame(tp0, columns=['d1', 'd2'])
    df_tp0['samples'] = 0

    df_tp1 = pd.DataFrame(tp1, columns=['d1', 'd2'])
    df_tp1['samples'] = 1

    df_tp2 = pd.DataFrame(tp2, columns=['d1', 'd2'])
    df_tp2['samples'] = 2

    df_tp3 = pd.DataFrame(tp3, columns=['d1', 'd2'])
    df_tp3['samples'] = 3

    df_tp4 = pd.DataFrame(tp4, columns=['d1', 'd2'])
    df_tp4['samples'] = 4

    # Concatenate the dataframes
    df = pd.concat([df_tp0, df_tp1, df_tp2, df_tp3, df_tp4], ignore_index=True)
    return df

def make_blob_death_example(n_pts_per_bin=100, seed=42):
    """
    Creates a 3-timepoint dataset of Gaussian blobs to test mass death explicitly.
    T0: Blob A (y=1) and Blob B (y=-1)
    T1: Blob A (y=1) and Blob B (y=-1), shifted right
    T2: Blob A (y=1), shifted right. Blob B is GONE.
    """
    np.random.seed(seed)
    
    # T0: Blob A (Top) and Blob B (Bottom)
    blob_a_t0 = np.random.randn(n_pts_per_bin, 2) * 0.15 + np.array([-1.0, 1.0])
    blob_b_t0 = np.random.randn(n_pts_per_bin, 2) * 0.15 + np.array([-1.0, -1.0])
    
    # T1: Blob A (shifted right) and Blob B (shifted right)
    blob_a_t1 = np.random.randn(n_pts_per_bin, 2) * 0.15 + np.array([0.0, 1.0])
    blob_b_t1 = np.random.randn(n_pts_per_bin, 2) * 0.15 + np.array([0.0, -1.0])
    
    # T2: Blob A (shifted further right). Blob B is GONE!
    blob_a_t2 = np.random.randn(n_pts_per_bin, 2) * 0.15 + np.array([1.0, 1.0])
    
    df_tp0 = pd.DataFrame(np.vstack([blob_a_t0, blob_b_t0]), columns=['d1', 'd2'])
    df_tp0['samples'] = 0.0
    
    df_tp1 = pd.DataFrame(np.vstack([blob_a_t1, blob_b_t1]), columns=['d1', 'd2'])
    df_tp1['samples'] = 1.0
    
    df_tp2 = pd.DataFrame(blob_a_t2, columns=['d1', 'd2'])
    df_tp2['samples'] = 2.0
    
    df = pd.concat([df_tp0, df_tp1, df_tp2], ignore_index=True)
    return df

def make_drastic_dying_example(n_pts_per_bin=50, seed=42):
    """
    Creates a drastically separated dataset where the bottom branch is far away.
    Top branch: y=0. Bottom branch: y=15 (dies after t=2).
    """
    import numpy as np
    import pandas as pd
    np.random.seed(seed)
    data = []
    
    for t in range(5):
        # Top branch
        x_top = np.random.normal(t * 2, 0.4, n_pts_per_bin)
        y_top = np.random.normal(0, 0.4, n_pts_per_bin)
        for i in range(n_pts_per_bin):
            data.append({'d1': x_top[i], 'd2': y_top[i], 'samples': str(t)})
            
        # Bottom branch
        if t <= 2:
            x_bottom = np.random.normal(t * 2, 0.4, n_pts_per_bin)
            y_bottom = np.random.normal(15, 0.4, n_pts_per_bin)
            for i in range(n_pts_per_bin):
                data.append({'d1': x_bottom[i], 'd2': y_bottom[i], 'samples': str(t)})
                
    return pd.DataFrame(data, columns=['d1', 'd2', 'samples'])
