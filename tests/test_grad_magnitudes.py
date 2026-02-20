import sys, os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import make_dying_example_unif
from mioflow import TimeSeriesDataset, ODEFunc, GrowthRateModel, train_mioflow, infer, ot_loss
from sklearn.preprocessing import StandardScaler

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
    scaled_data.append((torch.tensor(scaler.transform(x), dtype=torch.float32), t))

X_start, t_start = scaled_data[0]
X_end, _ = scaled_data[1]

growth_model = GrowthRateModel(input_dim=2, hidden_dim=32, use_time=True)

# forward
t_tensor = torch.full((X_start.size(0), 1), t_start, dtype=torch.float32)
source_mass = growth_model(X_start, t_tensor)

print("source_mass max:", source_mass.max().item(), "min:", source_mass.min().item())

loss_ot = ot_loss(X_start, X_end, source_mass=source_mass)
loss_l1 = torch.mean(torch.abs(source_mass - 1.0))

print("OT loss:", loss_ot.item())
print("L1 loss:", loss_l1.item())

# grads
# let's see grad of source_mass wrt OT loss
source_mass.retain_grad()
loss_ot.backward(retain_graph=True)
print("Max abs grad OT on source_mass:", source_mass.grad.abs().max().item())

growth_model.zero_grad()
source_mass.grad.zero_()

loss_l1.backward(retain_graph=True)
print("Max abs grad L1 on source_mass:", source_mass.grad.abs().max().item())

