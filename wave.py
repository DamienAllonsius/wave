import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot_simualtion(X, Y, Z):
    fig = plt.figure()

    sh_z = np.shape(Z)

    for t in range(sh_z[2]):
        ax = fig.gca(projection='3d')
        # Customize the z axis.
        ax.set_zlim(0, 10)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        ax.plot_surface(X, Y, Z[:, :, t], cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.pause(0.05)
        plt.clf()

    plt.show()

def t_x_p(Zt):
    _Zt = np.zeros(np.shape(Zt))
    _Zt[:, :-1] = Zt[:, 1:]
    _Zt[:, -1] = _Zt[:, -2]
    return _Zt

def t_y_p(Zt):
    _Zt = np.zeros(np.shape(Zt))
    _Zt[:-1, :] = Zt[1:, :]
    _Zt[-1, :] = _Zt[-2, :]
    return _Zt

def t_x_m(Zt):
    _Zt = np.zeros(np.shape(Zt))
    _Zt[:, 1:] = Zt[:, :-1]
    _Zt[:, 0] = _Zt[:, 1]
    return _Zt

def t_y_m(Zt):
    _Zt = np.zeros(np.shape(Zt))
    _Zt[1:, :] = Zt[:-1, :]
    _Zt[0, :] = _Zt[1, :]
    return _Zt

def L(Zt):
    Lx = 1/dx**2 * (t_x_p(Zt) - 2 * Zt + t_x_m(Zt))
    Ly = 1/dy**2 * (t_y_p(Zt) - 2 * Zt + t_y_m(Zt))
    return Lx + Ly

def compute_wave_equation(X, Y, Z0):
    Z0[:,0] = Z0[:,1]
    Z0[:,-1] = Z0[:,-2]
    Z0[0,:] = Z0[1,:]
    Z0[-1,:] = Z0[-2,:]
    Z = np.ndarray((Nx, Ny, NT))
    Z[:, :, 0] = Z0
    Z[:, :, 1] = Z0  # initial velocity = 0

    for t in range(1, NT-1):
        Z[:, :, t+1] = (dt ** 2) * (0.5 * L(Z[:, : , t]) + 0.5 * L(Z[:, : , t-1])) + 2 * Z[:, :, t] - Z[:, :, t-1]

        Z[0, :, t+1] = Z[1, :, t+1]
        Z[:, 0, t+1] = Z[:, 1, t+1]
        Z[-1, :, t+1] = Z[-2, :, t+1]
        Z[:, -1, t+1] = Z[1, :, t+1]

    return Z

x_min = -1
x_max = 1
y_min = -1
y_max = 1
Nx = 300
Ny = Nx
NT = 100
T_max = 1

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

Z = compute_wave_equation(X, Y, Z0)

plot_simualtion(X, Y, Z)
