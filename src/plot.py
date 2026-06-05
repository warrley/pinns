import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

def plot_solution(model, nx=256, nt=100):
    model.eval()
    t = torch.linspace(0, 1, nt)
    x = torch.linspace(-1, 1, nx)
    T, X = torch.meshgrid(t, x, indexing='ij')
    with torch.no_grad():
        U = model(
            T.reshape(-1,1),
            X.reshape(-1,1)
        )
    U = U.reshape(nt, nx).cpu().numpy()
    plt.figure(figsize=(10,6))
    plt.imshow(
        U.T,
        extent=[0,1,-1,1],
        origin='lower',
        aspect='auto',
        cmap='jet'
    )
    plt.colorbar(label='u(t,x)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('PINN Solution')
    plt.show()

def plot_1d_heatmap(U, title="1D Heat Equation - PINN", filename="spatiotemporal_heatmap.png"):
    plt.figure(figsize=(10,6))
    plt.imshow(
        U,
        extent=[0,1,0,1],
        origin='lower',
        aspect='auto',
        cmap='jet'
    )
    plt.colorbar(label='Temperature')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(title)
    plt.savefig(filename)
    plt.show()

def plot_1d_surface(X, T, U, title="PINN Solution of Heat Equation", filename="spatiotemporal_surface.png"):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('Temperature')
    ax.set_title(title)
    surf = ax.plot_surface(
        X, T, U,
        cmap='jet'
    )
    plt.savefig(filename)

def plot_2d_comparison(X_np, Y_np, U_pred, U_exact, t_val, filename="comparison.png"):
    error = np.abs(U_exact - U_pred)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plots = [
        (U_pred, "PINN Prediction"),
        (U_exact, "Exact Analytical Solution"),
        (error, "Absolute Error")
    ]
    for i, (ax, (data, title)) in enumerate(zip(axes, plots)):
        if i < 2:
            contour = ax.contourf(
                X_np, Y_np, data, 
                levels=100, cmap='coolwarm', vmin=0, vmax=1
            )
        else:
            contour = ax.contourf(
                X_np, Y_np, data, 
                levels=100, cmap='Reds'
            )
        ax.set_title(title)
        ax.set_xlabel('x dimension')
        ax.set_ylabel('y dimension')
        fig.colorbar(contour, ax=ax)
    plt.suptitle(f'2D Heat Equation Comparison at t = {t_val}', fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_2d_interactive(U_pred, x_val, y_val, t_val):
    fig = go.Figure(data=[go.Surface(
        z=U_pred, 
        x=x_val, 
        y=y_val, 
        colorscale='jet' 
    )])
    fig.update_layout(
        title=f'2D Heat Equation at t = {t_val}',
        autosize=False,
        width=800, 
        height=600,
        scene=dict(
            xaxis_title='X (Space)', 
            yaxis_title='Y (Space)', 
            zaxis_title='Temperature (u)'
        )
    )
    fig.show()

def generate_2d_animation(model, device, resolution=100, total_frames=40, filename='temperature_evolution.gif'):
    from IPython.display import HTML
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_flat = X.reshape(-1, 1).to(device)
    y_flat = Y.reshape(-1, 1).to(device)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    def update_3d_frame(frame):
        ax.clear()
        t_val = frame / (total_frames - 1)
        t_flat = torch.full_like(x_flat, t_val).to(device)
        model.eval()
        with torch.no_grad():
            u_pred = model(t_flat, x_flat, y_flat)
        U = u_pred.reshape(resolution, resolution).cpu().numpy()
        surf = ax.plot_surface(
            X.numpy(), Y.numpy(), U, 
            cmap='coolwarm', 
            rstride=1, cstride=1, linewidth=0, antialiased=False,
            vmin=0, vmax=1
        )
        ax.set_zlim(0, 1) 
        ax.set_xlabel('X dimension')
        ax.set_ylabel('Y dimension')
        ax.set_zlabel('Temperature (u)')
        ax.set_title(f'Smooth 3D Heat Diffusion: t = {t_val:.2f}')
        return surf,

    ani = animation.FuncAnimation(fig, update_3d_frame, frames=total_frames, blit=False)
    ani.save(filename, writer='pillow', fps=10)
    print(f"GIF successfully saved to {filename}!")
    plt.close() 
    return HTML(ani.to_jshtml())
