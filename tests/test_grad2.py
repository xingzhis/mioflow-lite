import torch
import ot

mu = torch.tensor([0.5, 0.5], requires_grad=True)
nu = torch.tensor([0.5, 0.5])
M = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

# reg = 0.1
loss = ot.sinkhorn2(mu, nu, M, reg=0.1)
loss.backward()

print("Gradient of mu (sinkhorn2):", mu.grad)
