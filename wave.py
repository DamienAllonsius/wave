import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot_simualtion(X, Y, Z):
    fig = plt.figure()

    for t in range(NT):
        ax = fig.gca(projection='3d')
        # Customize the z axis.
        ax.set_zlim(0, 10)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        ax.plot_surface(X, Y, Z[t, :, :], cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.pause(0.01)
        plt.clf()

    plt.show()

def t_x_p(Zt):
    return Zt[1:-1, 2:]

def t_y_p(Zt):
    return Zt[2:, 1:-1]

def t_x_m(Zt):
    return Zt[1:-1, :-2]

def t_y_m(Zt):
    return Zt[:-2, 1:-1]

def L(Zt):
    Lx = 1/dx**2 * (t_x_p(Zt) - 2 * Zt[1:-1, 1:-1] + t_x_m(Zt))
    Ly = 1/dy**2 * (t_y_p(Zt) - 2 * Zt[1:-1, 1:-1] + t_y_m(Zt))
    return Lx + Ly

def set_boundary_conditions(Z):
    Z[:,0] = Z[:,1]
    Z[:,-1] = Z[:,-2]
    Z[0,:] = Z[1,:]
    Z[-1,:] = Z[-2,:]
    return Z

def compute_wave_solution(Z0):
    Z0 = set_boundary_conditions(Z0)
    Z = np.zeros((NT, Nx, Ny))

    Z[0, :, :] = Z0
    Z[1, :, :] = Z0  # initial velocity = 0

    for t in range(1, NT-1):
        Z[t+1, 1:-1, 1:-1] = (dt ** 2) * (0.5 * L(Z[t, :, :]) + 0.5 * L(Z[t-1, :, :])) + 2 * Z[t, 1:-1, 1:-1] - Z[t-1, 1:-1, 1:-1]
        Z[t+1, :, :] = set_boundary_conditions(Z[t+1, :, :])

    return Z

x_min = -1
x_max = 1
y_min = -1
y_max = 1
Nx = 100
Ny = Nx
NT = 1000
T_max = 2

# Uniform discretization w.r.t x
x = np.linspace(x_min, x_max, Nx)
dx = x[1] - x[0]

# Uniform discretization w.r.t y
y = np.linspace(y_min, y_max, Ny)
dy = y[1] - y[0]

# Uniform discretization w.r.t time
dt = T_max / NT

# mesh x and y
X, Y = np.meshgrid(x, y)

# minus laplacian
# m_Lapl_x = (1 / dx ** 2) * np.diag(2 * np.ones(Nx)) - np.diag(np.ones(Nx - 1), -1) - np.diag(np.ones(Nx - 1), 1)

Z0 = 10 * np.exp(- 10*X ** 2) * np.exp(- 10* Y ** 2)

Z = compute_wave_solution(Z0)

plot_simualtion(X, Y, Z)
