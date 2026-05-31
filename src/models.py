import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        
        layers = []
        # Primeira camada
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Camadas ocultas
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # Camada de saída
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, *inputs):
        # inputs can be (t, x) for 1D or (t, x, y) for 2D
        x = torch.cat(inputs, dim=1)
        return self.net(x)