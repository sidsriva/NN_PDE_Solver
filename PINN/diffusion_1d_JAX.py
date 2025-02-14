import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from flax import linen as nn
from jax import grad, jit, vmap

# Set JAX to use GPU (MPS for Mac, CUDA for Nvidia)
# jax.devices('METAL')
jax.devices('cpu')
# Define the Neural Network using Flax

class PINN(nn.Module):
    layer_sizes: list  # List of layer sizes (e.g., [2, 50, 50, 50, 1])

    def setup(self):
        """Define layers in the model."""
        self.layers = [nn.Dense(size) for size in self.layer_sizes[:-1]]
        self.output_layer = nn.Dense(self.layer_sizes[-1])  # Output layer

    def __call__(self, x, t):
        """Forward pass"""
        inputs = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers:
            inputs = nn.tanh(layer(inputs))
        return self.output_layer(inputs)

# PDE parameters
alpha = 0.01

# Initialize model
layers = [2, 50, 50, 50, 1]  # Increased network depth
model = PINN(layers)

# Initialize parameters
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, 1)), jnp.ones((1, 1)))  # Init with dummy data

# Define the PDE loss function using JAX's autodiff
def pde_loss(params, x, t):
    def model_output_scalar(x, t):
        return model.apply(params, x, t).squeeze()  # Ensure scalar output

    # First derivatives
    u_t = jax.vmap(grad(model_output_scalar, argnums=1))(x, t)  # du/dt
    u_x = jax.vmap(grad(model_output_scalar, argnums=0))(x, t)  # du/dx

    # Second derivative
    def second_derivative(x, t):
        return grad(model_output_scalar, argnums=0)(x, t).squeeze()  # Ensure scalar

    u_xx = jax.vmap(grad(second_derivative, argnums=0))(x, t)  # d²u/dx²

    residual = u_t - alpha * u_xx
    return jnp.mean(residual**2)  # Ensure scalar loss

# Define boundary and initial condition losses
def boundary_loss(params, x_bc, t_bc):
    return jnp.mean(model.apply(params, x_bc, t_bc) ** 2)

def initial_loss(params, x_ic, t_ic, u_ic):
    return jnp.mean((model.apply(params, x_ic, t_ic) - u_ic) ** 2)

# JIT compile the loss functions for faster execution
pde_loss = jit(pde_loss)
boundary_loss = jit(boundary_loss)
initial_loss = jit(initial_loss)

# Generate training data
N_f = 10000  # Collocation points for PDE residual
N_bc = 200   # Boundary points (increased for better constraint enforcement)
N_ic = 200   # Initial condition points

key_x, key_t = jax.random.split(key)
x_f = jax.random.uniform(key_x, (N_f, 1))
t_f = jax.random.uniform(key_t, (N_f, 1))

x_bc = jnp.concatenate([jnp.zeros((N_bc, 1)), jnp.ones((N_bc, 1))], axis=0)
t_bc = jax.random.uniform(key, (N_bc * 2, 1))

x_ic = jax.random.uniform(key, (N_ic, 1))
t_ic = jnp.zeros((N_ic, 1))
u_ic = jnp.sin(np.pi * x_ic)

# Optimizer with Learning Rate Scheduler
learning_rate = 1e-3
scheduler = optax.exponential_decay(init_value=learning_rate, transition_steps=5000, decay_rate=0.98)
optimizer = optax.adam(scheduler)
opt_state = optimizer.init(params)

# Training loop
epochs = 5000  # Increase training time
batch_size = 10000  # Increase batch size for stability

@jit
def update(params, opt_state, x_f_batch, t_f_batch, epoch):
    """Single training step"""
    loss_pde = pde_loss(params, x_f_batch, t_f_batch)
    loss_bc = boundary_loss(params, x_bc, t_bc)
    loss_ic = initial_loss(params, x_ic, t_ic, u_ic)
    
    total_loss = loss_pde + loss_bc + loss_ic
    
    # JIT only gradient calculation, NOT `epoch`
    def compute_gradients(params):
        grads = jax.grad(lambda p: loss_pde + loss_bc + loss_ic)(params)
        return grads

    grads = compute_gradients(params)

    # Compute gradient norm correctly
    grad_values, _ = jax.tree_util.tree_flatten(grads)
    grad_norm = jnp.linalg.norm(jnp.array([jnp.linalg.norm(g) for g in grad_values]))

    # jax.debug.print("Epoch {} | Loss: {:.6f} | Grad Norm: {:.6f}", jnp.array(epoch), total_loss, grad_norm)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, total_loss

loss_history = []
for epoch in range(epochs):
    # Sample a batch of collocation points
    idx = np.random.randint(0, N_f, batch_size)
    x_f_batch, t_f_batch = x_f[idx], t_f[idx]
    
    params, opt_state, loss = update(params, opt_state, x_f_batch, t_f_batch, epoch)
    loss_history.append(loss)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("PINN Training Loss")
plt.legend()
plt.show()

# Generate animation of u(x,t) evolving over time
x_test = jnp.linspace(0, 1, 100).reshape(-1, 1)
time_steps = jnp.linspace(0, 1, 50)  # 50 time snapshots

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("PINN Solution of the Diffusion Equation Over Time")

line, = ax.plot([], [], lw=2)

def update_frame(frame):
    t_test = jnp.full_like(x_test, time_steps[frame])
    u_pred = model.apply(params, x_test, t_test)
    line.set_data(x_test, u_pred)
    ax.set_title(f"PINN Solution at t = {time_steps[frame]:.2f}")
    return line,

ani = animation.FuncAnimation(fig, update_frame, frames=len(time_steps), interval=100)

# Save animation as a GIF
ani.save("./results/pinn_diffusion_1d_JAX.gif", writer="pillow", fps=10)

plt.show()