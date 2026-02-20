import numpy as np
import ot

def test_mm_unbalanced():
    # 2 source points, 1 target point
    # Source 0 is close (good), Source 1 is far (bad)
    x0 = np.array([[0.0, 0.0], [10.0, 0.0]])
    x1 = np.array([[0.0, 0.0]])
    
    m = ot.unif(len(x0)) # [0.5, 0.5]
    n = ot.unif(len(x1)) # [1.0]
    
    M = ot.dist(x0, x1) # Squared euclidean distance
    print("Cost matrix M:\n", M)
    
    # Test different reg_m and div
    for div in ['l2', 'kl']:
        for reg_m in [1.0, 5.0, 10.0, 50.0, 100.0]:
            try:
                # reg_m is a scalar or tuple (alpha, beta) for source/target marginal penalization
                plan = ot.unbalanced.mm_unbalanced(m, n, M, reg_m, div=div)
                
                # Marginal mass of each source point
                source_marginal = plan.sum(axis=1)
                
                # Multiply by N to keep scale relative to 1.0
                normalized_gr = source_marginal * len(x0)
                
                print(f"div={div:2s}, reg_m={reg_m:5.1f} -> {normalized_gr}")
            except Exception as e:
                print(f"Failed for div={div}, reg_m={reg_m}: {e}")

if __name__ == "__main__":
    test_mm_unbalanced()
