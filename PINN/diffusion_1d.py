import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Check for MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Neural Network
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())

    def forward(self, x, t):
        inputs = torch.cat((x, t), dim=1)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

# PDE parameters
alpha = 0.01

# Loss function
def pde_loss(model, x, t):
    x.requires_grad = True
    t.requires_grad = True

    u = model(x, t)
    
    u_t = torch.autograd.grad(u, t, torch.ones_like(u).to(device), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u).to(device), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x).to(device), create_graph=True)[0]

    residual = u_t - alpha * u_xx
    return torch.mean(residual**2)

# Boundary and Initial Condition Loss
def boundary_loss(model, x_bc, t_bc):
    return torch.mean(model(x_bc, t_bc) ** 2)

def initial_loss(model, x_ic, t_ic, u_ic):
    return torch.mean((model(x_ic, t_ic) - u_ic) ** 2)

# Training data (Move to MPS)
N_f = 10000  # Collocation points for PDE residual
N_bc = 100   # Boundary points
N_ic = 100   # Initial condition points

x_f = torch.rand(N_f, 1, device=device)
t_f = torch.rand(N_f, 1, device=device)

x_bc = torch.cat((torch.zeros(N_bc, 1), torch.ones(N_bc, 1)), dim=0).to(device)
t_bc = torch.rand(N_bc * 2, 1, device=device)

x_ic = torch.rand(N_ic, 1, device=device)
t_ic = torch.zeros(N_ic, 1, device=device)
u_ic = torch.sin(np.pi * x_ic).to(device)

# Initialize PINN model and move to MPS
layers = [2, 20, 20, 20, 1]
model = PINN(layers).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    loss_pde = pde_loss(model, x_f, t_f)
    loss_bc = boundary_loss(model, x_bc, t_bc)
    loss_ic = initial_loss(model, x_ic, t_ic, u_ic)
    
    loss = loss_pde + loss_bc + loss_ic
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Generate animation of u(x,t) evolving over time
x_test = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
time_steps = torch.linspace(0, 1, 50, device=device)  # 50 time snapshots

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("PINN Solution of the Diffusion Equation Over Time")

line, = ax.plot([], [], lw=2)

def update(frame):
    t_test = torch.full_like(x_test, time_steps[frame])
    u_pred = model(x_test, t_test).detach().cpu().numpy()
    line.set_data(x_test.cpu().numpy(), u_pred)
    ax.set_title(f"PINN Solution at t = {time_steps[frame].item():.2f}")
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=100)

# Save animation as a video file
ani.save("./results/pinn_diffusion_1d.gif", writer="ffmpeg", fps=10)

plt.show()