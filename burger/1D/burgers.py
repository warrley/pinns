# Import required libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network architecture
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(2, layers[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.output_layer = nn.Linear(layers[-1], 1)
        self.activation = nn.Tanh()

    def forward(self, t, x):
        input_tensor = torch.cat([t, x], dim=1)
        input_tensor = self.activation(self.input_layer(input_tensor))
        for layer in self.hidden_layers:
            input_tensor = self.activation(layer(input_tensor))
        return self.output_layer(input_tensor)
        
# Define the physics-informed loss function
def pde_loss(model, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)

    u = model(t, x)

    # Compute partial derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # Define the PDE residual
    f = u_t + u * u_x - (0.01 / np.pi) * u_xx
    return f
def create_training_data():
    # Initial condition
    x = np.linspace(-1, 1, 256).reshape(-1, 1)
    t = np.zeros_like(x)
    u = -np.sin(np.pi * x)

    # Boundary conditions
    t_boundary = np.linspace(0, 1, 100).reshape(-1, 1)
    x_left = -np.ones_like(t_boundary)
    x_right = np.ones_like(t_boundary)
    u_left = np.zeros_like(t_boundary)
    u_right = np.zeros_like(t_boundary)

    # Collocation points
    t_f = np.random.uniform(0, 1, 10000).reshape(-1, 1)
    x_f = np.random.uniform(-1, 1, 10000).reshape(-1, 1)

    # Convert all to PyTorch tensors
    t_u = torch.tensor(t, dtype=torch.float32)
    x_u = torch.tensor(x, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    t_boundary = torch.tensor(t_boundary, dtype=torch.float32)
    x_left = torch.tensor(x_left, dtype=torch.float32)
    x_right = torch.tensor(x_right, dtype=torch.float32)
    u_left = torch.tensor(u_left, dtype=torch.float32)
    u_right = torch.tensor(u_right, dtype=torch.float32)
    t_f = torch.tensor(t_f, dtype=torch.float32)
    x_f = torch.tensor(x_f, dtype=torch.float32)

    return t_u, x_u, u, t_boundary, x_left, x_right, u_left, u_right, t_f, x_f

# Define the training loop
def train(model, optimizer, t_u, x_u, u, t_f, x_f, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Data loss
        u_pred = model(t_u, x_u)
        loss_u = torch.mean((u - u_pred) ** 2)

        # PDE residual loss
        f_pred = pde_loss(model, t_f, x_f)
        loss_f = torch.mean(f_pred ** 2)

        # Total loss
        loss = loss_u + loss_f
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Loss_u: {loss_u.item()}, Loss_f: {loss_f.item()}")
            
# Plotting function
def plot_results(model):
    t = np.linspace(0, 1, 100)
    x = np.linspace(-1, 1, 256)
    T, X = np.meshgrid(t, x)

    t_test = torch.tensor(T.flatten().reshape(-1, 1), dtype=torch.float32)
    x_test = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32)
    u_pred = model(t_test, x_test).detach().numpy().reshape(256, 100)

    plt.figure(figsize=(10, 6))
    plt.imshow(u_pred, extent=[0, 1, -1, 1], origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label='u(t, x)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Predicted solution u(t, x)')
    plt.show()
   
# Main execution
if __name__ == "__main__":
    layers = [50, 50, 50, 50]
    model = PINN(layers)

    t_u, x_u, u, t_boundary, x_left, x_right, u_left, u_right, t_f, x_f = create_training_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, optimizer, t_u, x_u, u, t_f, x_f, epochs=5000)
    plot_results(model)
