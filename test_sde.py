"""
Simple test script to verify ODE and SDE functionality in mioflow.py
"""
import torch
import numpy as np
import sys
sys.path.insert(0, 'src')

from mioflow import ODEFunc, SDEFunc, TimeSeriesDataset, train_mioflow, infer

def generate_toy_data():
    """Generate simple toy time series data for testing"""
    np.random.seed(42)
    time_series_data = []
    
    # Create 3 time points with simple dynamics
    for t in range(3):
        # Generate points that move linearly
        n_points = 100
        x = np.random.randn(n_points, 2) + t * 0.5
        time_series_data.append((x, float(t)))
    
    return time_series_data

def test_ode():
    """Test ODE model"""
    print("=" * 60)
    print("Testing ODE Model")
    print("=" * 60)
    
    # Create toy data
    data = generate_toy_data()
    dataset = TimeSeriesDataset(data)
    
    # Create ODE model
    model = ODEFunc(input_dim=2, hidden_dim=32)
    print(f"✓ Created ODEFunc model")
    
    # Test inference
    x0 = torch.randn(10, 2)
    t_seq = torch.tensor([0.0, 1.0, 2.0])
    trajectory = infer(x0, model, t_seq)
    print(f"✓ Inference works: trajectory shape = {trajectory.shape}")
    
    # Test training (just 2 epochs to verify it runs)
    history = train_mioflow(
        model=model,
        dataset=dataset,
        num_epochs=2,
        batch_size=50,
        learning_rate=1e-3,
        lambda_ot=1.0,
        lambda_density=0.1,
        lambda_energy=0.01,
        device='cpu'
    )
    print(f"✓ Training works: final loss = {history['total_loss'][-1]:.4f}")
    print()

def test_sde():
    """Test SDE model"""
    print("=" * 60)
    print("Testing SDE Model")
    print("=" * 60)
    
    # Create toy data
    data = generate_toy_data()
    dataset = TimeSeriesDataset(data)
    
    # Create SDE model
    model = SDEFunc(input_dim=2, hidden_dim=32)
    print(f"✓ Created SDEFunc model")
    print(f"  - noise_type: {model.noise_type}")
    print(f"  - sde_type: {model.sde_type}")
    
    # Test inference
    x0 = torch.randn(10, 2)
    t_seq = torch.tensor([0.0, 1.0, 2.0])
    trajectory = infer(x0, model, t_seq, dt=0.01)
    print(f"✓ Inference works: trajectory shape = {trajectory.shape}")
    
    # Test training (just 2 epochs to verify it runs)
    history = train_mioflow(
        model=model,
        dataset=dataset,
        num_epochs=2,
        batch_size=50,
        learning_rate=1e-3,
        lambda_ot=1.0,
        lambda_density=0.1,
        lambda_energy=0.01,
        sde_dt=0.01,
        device='cpu'
    )
    print(f"✓ Training works: final loss = {history['total_loss'][-1]:.4f}")
    print()

if __name__ == '__main__':
    print("\nRunning MIOFlow Tests\n")
    
    try:
        test_ode()
        test_sde()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

