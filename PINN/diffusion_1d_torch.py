import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Check for MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
jit_model = True

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
layers = [2, 50, 50, 50, 1]
model = PINN(layers).to(device)
if jit_model == True:
    model = torch.jit.script(model)  # JIT-compile only the forward pass

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 2000
batch_size = 512  # Adjust as needed
loss_history = []
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Sample a batch of collocation points
    idx = torch.randint(0, N_f, (batch_size,))
    x_f_batch, t_f_batch = x_f[idx], t_f[idx]
    
    loss_pde = pde_loss(model, x_f_batch, t_f_batch)
    loss_bc = boundary_loss(model, x_bc, t_bc)
    loss_ic = initial_loss(model, x_ic, t_ic, u_ic)
    
    loss = loss_pde + loss_bc + loss_ic
    loss_history.append(loss.item())
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Plot loss history
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("PINN Training Loss Over Time")
plt.legend()
plt.savefig("./results/pinn_diffusion_1d_torch_loss.pdf")
plt.show()

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
ani.save("./results/pinn_diffusion_1d_torch.gif", writer="ffmpeg", fps=10)

plt.show()