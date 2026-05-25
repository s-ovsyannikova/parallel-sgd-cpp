import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

def plot_speedup_surface(filename='speedup_data.csv'):
    df = pd.read_csv(filename)

    X = df['BatchSize'].values
    Y = df['NumThreads'].values
    Z = df['Speedup'].values

    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    
    X_line = np.unique(X)
    Y_line = np.unique(Y)
    
    X_grid, Y_grid = np.meshgrid(X_line, Y_line)

    Z_grid = griddata((X, Y), Z, (X_grid, Y_grid), method='linear')
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(X_grid, Y_grid, Z_grid, 
                              cmap='viridis', 
                              linewidth=0, 
                              antialiased=False,
                              alpha=0.7)

    ax.scatter(X, Y, Z, color='r', marker='o', s=10, label='Original results')

    ax.set_xlabel('Batch size', fontsize=10)
    ax.set_ylabel('Number of threads', fontsize=12)
    ax.set_zlabel('Speedup', fontsize=12)
    ax.set_title('3D-surface, speedup (interpolation)', fontsize=14)

    fig.colorbar(surface, shrink=0.5, aspect=5, label='Speedup')
    
    ax.view_init(elev=30, azim=145)
    ax.set_xticks(X_line)
    ax.set_yticks(Y_line)
    
    plt.show()

if __name__ == "__main__":
    plot_speedup_surface()