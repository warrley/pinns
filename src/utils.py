import torch

def get_device():
    """Configura e retorna o device (CUDA ou CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.init()
        # Otimiza multiplicações de matriz em GPUs Ampere+
        torch.set_float32_matmul_precision('high')
    return device

def derivative(y, x):
    """Calcula a derivada de y em relação a x."""
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    return grad
