import torch
import ot

source_mass = torch.tensor([1.0, 1.0], requires_grad=True) # Top dot, Bottom dot
target_mass = torch.tensor([1.0, 1.0])

# Normalization: divide by number of points so that base mass = 1.0 means uniform distribution
mu = source_mass / source_mass.size()[0]
nu = target_mass / target_mass.size()[0]

# M: Top dot to Top target is cheap (0.1), Bottom dot to Top target is expensive (10.0)
M = torch.tensor([[0.1, 0.1], [10.0, 10.0]])

# Unbalanced Sinkhorn
loss_ot = ot.unbalanced.sinkhorn_unbalanced2(mu, nu, M, reg=0.1, reg_m=1.0)
loss_l1 = 0.1 * torch.mean(torch.abs(source_mass - 1.0))
loss = loss_ot + loss_l1

loss.backward()

optimizer = torch.optim.Adam([source_mass], lr=0.1)
for i in range(200):
    optimizer.zero_grad()
    s_mass = torch.nn.functional.softplus(source_mass) + 1e-4
    mu = s_mass / s_mass.size()[0]
    nu = target_mass / target_mass.size()[0]
    loss_ot = ot.unbalanced.sinkhorn_unbalanced2(mu, nu, M, reg=0.5, reg_m=1.0)
    loss_l1 = 0.05 * torch.mean(torch.abs(s_mass - 1.0))
    loss = loss_ot
    loss.backward()
    optimizer.step()

final_mass = torch.nn.functional.softplus(source_mass) + 1e-4
print("Final mass:", final_mass.detach())
