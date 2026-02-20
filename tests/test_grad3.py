import torch
import ot

# Simulate Joint Training Scenario
source_mass = torch.tensor([1.0, 1.0], requires_grad=True) # Top dot, Bottom dot
target_mass = torch.tensor([1.0, 1.0])

# Normalization
mu = source_mass / source_mass.sum()
nu = target_mass / target_mass.sum()

# M: Top dot to Top target is cheap (0.1), Bottom dot to Top target is expensive (10.0)
M = torch.tensor([[0.1, 0.1], [10.0, 10.0]])

# Forward
loss_ot = ot.sinkhorn2(mu, nu, M, reg=0.1, method='sinkhorn_log', numItermax=500)
loss_l1 = 0.1 * torch.mean(torch.abs(source_mass - 1.0))
loss = loss_ot + loss_l1

loss.backward()

print("OT Loss:", loss_ot.item())
print("L1 Loss:", loss_l1.item())
print("Gradient of source_mass:", source_mass.grad)

# What if we optimizer it?
optimizer = torch.optim.Adam([source_mass], lr=0.1)
for i in range(100):
    optimizer.zero_grad()
    mu = torch.nn.functional.softplus(source_mass) + 1e-4
    mu = mu / mu.sum()
    nu = target_mass / target_mass.sum()
    loss_ot = ot.sinkhorn2(mu, nu, M, reg=0.1, method='sinkhorn_log', numItermax=500)
    loss_l1 = 0.1 * torch.mean(torch.abs(torch.nn.functional.softplus(source_mass) + 1e-4 - 1.0))
    loss = loss_ot + loss_l1
    loss.backward()
    optimizer.step()

final_mass = torch.nn.functional.softplus(source_mass) + 1e-4
print("Final mass:", final_mass.detach())
