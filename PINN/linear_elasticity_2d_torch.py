import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# Check for MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
jit_model = True  # Enable JIT compilation

# Define the Neural Network for 2D Linear Elasticity
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())

    def forward(self, x, y):
        """Forward pass"""

        if x.dim() == 1:
            x = x.view(-1, 1)  # Ensure shape (N, 1)
        if y.dim() == 1:
            y = y.view(-1, 1)  # Ensure shape (N, 1)

        inputs = torch.cat((x, y), dim=1)
        for layer in self.layers:
            inputs = layer(inputs)
        u_x = inputs[:, 0:1]  # Displacement in x
        u_y = inputs[:, 1:2]  # Displacement in y
        return u_x, u_y  # Return displacements separately

# Material Properties (Steel-like example)
E = 200.0  # Young's modulus
nu = 0.3   # Poisson's ratio
lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))  # First Lame parameter
mu = E / (2 * (1 + nu))  # Shear modulus

# PDE Loss Function (Linear Elasticity Equations)
def elasticity_loss(model, x, y):
    x.requires_grad = True
    y.requires_grad = True

    u_x, u_y = model(x, y)

    # Compute gradients
    u_x_x = torch.autograd.grad(u_x, x, torch.ones_like(u_x).to(device), create_graph=True)[0]
    u_x_y = torch.autograd.grad(u_x, y, torch.ones_like(u_x).to(device), create_graph=True)[0]
    u_y_x = torch.autograd.grad(u_y, x, torch.ones_like(u_y).to(device), create_graph=True)[0]
    u_y_y = torch.autograd.grad(u_y, y, torch.ones_like(u_y).to(device), create_graph=True)[0]

    # Compute strain tensor components
    epsilon_xx = u_x_x
    epsilon_yy = u_y_y
    epsilon_xy = 0.5 * (u_x_y + u_y_x)

    # Compute stress tensor using Hooke's Law
    sigma_xx = lambda_ * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_xx
    sigma_yy = lambda_ * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_yy
    sigma_xy = 2 * mu * epsilon_xy

    # Compute equilibrium residuals (assuming no external force)
    sigma_xx_x = torch.autograd.grad(sigma_xx, x, torch.ones_like(sigma_xx).to(device), create_graph=True)[0]
    sigma_xy_y = torch.autograd.grad(sigma_xy, y, torch.ones_like(sigma_xy).to(device), create_graph=True)[0]

    sigma_yy_y = torch.autograd.grad(sigma_yy, y, torch.ones_like(sigma_yy).to(device), create_graph=True)[0]
    sigma_xy_x = torch.autograd.grad(sigma_xy, x, torch.ones_like(sigma_xy).to(device), create_graph=True)[0]

    # Residual equations (force balance)
    residual_x = sigma_xx_x + sigma_xy_y  # ∂σ_xx/∂x + ∂σ_xy/∂y = 0
    residual_y = sigma_yy_y + sigma_xy_x  # ∂σ_yy/∂y + ∂σ_xy/∂x = 0

    return torch.mean(residual_x ** 2 + residual_y ** 2)

# Boundary Condition Loss
def boundary_loss(model):
    """
    Apply different boundary conditions on each of the 4 boundaries
        Clamped left boundary (x = 0).
	    Prescribed displacement on the right boundary (x = 1).
	    Fixed bottom boundary (y = 0).
	    Shear force on the top boundary (y = 1).
	    Zero traction on the circle (Neumann BC).
    """

    N_per_side =10
    N_circle = 50    # Number of points on the circular boundary

    # Left boundary (x=0, 0 ≤ y ≤ 1)
    x_left = torch.zeros(N_per_side, 1, device=device)
    y_left = torch.rand(N_per_side, 1, device=device)
    n_x_left = -1*torch.ones(N_per_side, 1, device=device)
    n_y_left =   torch.zeros(N_per_side, 1, device=device)

    # Right boundary (x=1, 0 ≤ y ≤ 1)
    x_right = torch.ones(N_per_side, 1, device=device)
    y_right = torch.rand(N_per_side, 1, device=device)
    n_x_right = torch.ones(N_per_side, 1, device=device)
    n_y_right = torch.zeros(N_per_side, 1, device=device)

    # Bottom boundary (0 ≤ x ≤ 1, y=0)
    x_bottom = torch.rand(N_per_side, 1, device=device)
    y_bottom = torch.zeros(N_per_side, 1, device=device)
    n_x_bottom = torch.zeros(N_per_side, 1, device=device)
    n_y_bottom = -1*torch.ones(N_per_side, 1, device=device)

    # Top boundary (0 ≤ x ≤ 1, y=1)
    x_top = torch.rand(N_per_side, 1, device=device)
    y_top = torch.ones(N_per_side, 1, device=device)
    n_x_top = torch.zeros(N_per_side, 1, device=device)
    n_y_top = torch.ones(N_per_side, 1, device=device)

    # Generate Points for Circular Boundary at Center (Zero Traction)

    circle_radius = 0.2
    circle_center = torch.tensor([0.5, 0.5], device=device)
    theta = torch.linspace(0, 2 * np.pi, N_circle, device=device).reshape(-1, 1)  # Angles for circle

    x_circle = circle_center[0] + circle_radius * torch.cos(theta)
    y_circle = circle_center[1] + circle_radius * torch.sin(theta)
    n_x_circle = -torch.cos(theta)
    n_y_circle = -torch.sin(theta)

    # Combine all boundary points
    x_bc = torch.cat([x_left, x_right, x_bottom, x_top, x_circle], dim=0).to(device)
    y_bc = torch.cat([y_left, y_right, y_bottom, y_top, y_circle], dim=0).to(device)
    n_x_bc = torch.cat([x_left, x_right, x_bottom, x_top, x_circle], dim=0).to(device)
    n_y_bc = torch.cat([x_left, x_right, x_bottom, x_top, y_circle], dim=0).to(device)

    # Forward simulation
    x_bc.requires_grad = True
    y_bc.requires_grad = True

    u_x, u_y = model(x_bc, y_bc)

    # Compute gradients
    u_x_x = torch.autograd.grad(u_x, x_bc, torch.ones_like(u_x).to(device), create_graph=True)[0]
    u_x_y = torch.autograd.grad(u_x, y_bc, torch.ones_like(u_x).to(device), create_graph=True)[0]
    u_y_x = torch.autograd.grad(u_y, x_bc, torch.ones_like(u_y).to(device), create_graph=True)[0]
    u_y_y = torch.autograd.grad(u_y, y_bc, torch.ones_like(u_y).to(device), create_graph=True)[0]

    # Compute strain tensor components
    epsilon_xx = u_x_x
    epsilon_yy = u_y_y
    epsilon_xy = 0.5 * (u_x_y + u_y_x)

    # Compute stress tensor using Hooke's Law
    sigma_xx = lambda_ * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_xx
    sigma_yy = lambda_ * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_yy
    sigma_xy = 2 * mu * epsilon_xy

    #Compute tractions
    traction_x = sigma_xx*n_x_bc + sigma_xy*n_y_bc 
    traction_y = sigma_xy*n_x_bc + sigma_yy*n_y_bc

    # Left boundary (x = 0) → traction free: σ.n = 0
    mask_left = (x_bc == 0)
    loss_left = torch.mean(traction_x[mask_left]**2 + traction_y[mask_left]**2)

    # Right boundary (x = 1) → traction free: σ.n = 0
    mask_right = (x_bc == 1)
    loss_right = torch.mean(traction_x[mask_right]**2 + traction_y[mask_right]**2)

    # Bottom boundary (y = 0) → Clamped: u_x = 0, u_y = 0
    mask_bottom = (y_bc == 0)
    loss_bottom = torch.mean(u_x[mask_bottom]**2 + u_y[mask_bottom]**2)

    # Top boundary (y = 1) → Shear traction applied: σ.n . (1,0) = f, σ.n . (0,1) = 0
    mask_top = (y_bc == 1)
    f=100
    loss_top = torch.mean((traction_x[mask_top]-f)**2 + traction_y[mask_top]**2)

    # Circular Boundary: Zero Traction Condition (Neumann BC)
    # σ.n = 0 ⇒ Surface traction = 0
    mask_circle = (torch.sqrt((x_bc - circle_center[0])**2 + (y_bc - circle_center[1])**2) - circle_radius).abs() < 1e-3
    loss_circle = torch.mean((traction_x[mask_circle]-f)**2 + traction_y[mask_circle]**2)

    # Total boundary loss
    return loss_left + loss_right + loss_bottom + loss_top + loss_circle


def sample_points_square_with_hole(N_f):
    circle_radius = 0.2
    circle_center = torch.tensor([0.5, 0.5], device=device)

    valid_points = []

    while len(valid_points) < N_f:
        # Sample candidate points uniformly in the square domain
        x_candidates = torch.rand(N_f, 1, device=device)
        y_candidates = torch.rand(N_f, 1, device=device)

        # Compute distance from the circle center
        distances = torch.sqrt((x_candidates - circle_center[0])**2 + (y_candidates - circle_center[1])**2)

        # Keep only points outside the circle
        mask_outside_circle = distances >= circle_radius
        x_valid = x_candidates[mask_outside_circle]
        y_valid = y_candidates[mask_outside_circle]

        # Add to the list of valid points
        valid_points.append((x_valid, y_valid))

        # Flatten the list and ensure we get exactly N_f points
        valid_points = [(x[:N_f], y[:N_f]) for x, y in valid_points]

    # Convert list of tensors to final collocation point tensors
    x_f, y_f = valid_points[0]
    return x_f, y_f

N_f = 10000  # Number of collocation points
x_f, y_f = sample_points_square_with_hole(N_f)
# Initialize PINN Model
layers = [2, 50, 50, 50, 2]  # Output is 2D (u_x, u_y)
model = PINN(layers).to(device)
if jit_model:
    model = torch.jit.script(model)  # JIT compile

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
epochs = 5000
batch_size = 512
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Sample a batch of collocation points
    idx = torch.randint(0, N_f, (batch_size,))
    x_f_batch, y_f_batch = x_f[idx], y_f[idx]

    loss_pde = elasticity_loss(model, x_f_batch, y_f_batch)
    loss_bc = boundary_loss(model)
    
    loss = loss_pde + loss_bc
    loss_history.append(loss.item())

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Plot Training Loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("PINN Training Loss for 2D Elasticity")
plt.legend()
plt.savefig('./results/linear_elasticity_2d_torch_loss.pdf')
# plt.show()

# Visualize Displacement Field
# Define Circular Hole Properties
circle_radius = 0.2
circle_center = np.array([0.5, 0.5])  # Center of the hole

x_test, y_test = torch.meshgrid(
    torch.linspace(0, 1, 50, device=device), 
    torch.linspace(0, 1, 50, device=device),
    indexing="ij"  # Ensure correct meshgrid ordering
)
x_test = x_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

u_x_pred, u_y_pred = model(x_test, y_test)
# u_x_pred = u_x_pred.detach().cpu().numpy().reshape(50, 50)
# u_y_pred = u_y_pred.detach().cpu().numpy().reshape(50, 50)

# Compute Distance from Circle Center
x_np = x_test.cpu().numpy().flatten()
y_np = y_test.cpu().numpy().flatten()
u_x_np = u_x_pred.cpu().detach().numpy().flatten()
u_y_np = u_y_pred.cpu().detach().numpy().flatten()
distances = np.sqrt((x_np - circle_center[0])**2 + (y_np - circle_center[1])**2)

# Mask Points Inside the Circular Hole
mask_outside_circle = distances >= circle_radius
x_filtered = x_np[mask_outside_circle]
y_filtered = y_np[mask_outside_circle]
u_x_filtered = u_x_np[mask_outside_circle]
u_y_filtered = u_y_np[mask_outside_circle]

# Plot displacement field
fig, ax = plt.subplots(figsize=(6, 6))
# X, Y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
ax.quiver(x_filtered, y_filtered, u_x_filtered, u_y_filtered, scale=20, scale_units='xy')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Predicted Displacement Field (u_x, u_y)")
plt.savefig('./results/linear_elasticity_2d_torch.pdf')
plt.show()