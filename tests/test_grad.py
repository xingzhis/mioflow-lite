import torch
import ot

mu = torch.tensor([0.5, 0.5], requires_grad=True)
nu = torch.tensor([0.5, 0.5])
M = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

loss = ot.emd2(mu, nu, M)
loss.backward()

print("Gradient of mu:", mu.grad)
