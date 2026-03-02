import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


a_vals = [2, -2]
dx = 0.1
x0, x1 = 0.0, 1.0
t0, t1 = 0.0, 10.0
dt = 0.05
Nt = int((t1 - t0) / dt)


def U0(x):
    return 2 * x**2


def f(x, t):
    return 3 * x


def U_left(t):
    return t**2


def U_right(t):
    return t**2 + 2


L_extra = 30
output_dir = "Results_6_lab"
os.makedirs(output_dir, exist_ok = True)
for a in a_vals:
    for region in ['half', 'rect']:
        for scheme in ['explicit', 'implicit']:
            print(f"Computing: a={a}, region={region}, scheme={scheme}")
            if region == 'rect':
                x = np.linspace(x0, x1, int((x1 - x0) / dx) + 1)
                Nx = len(x)
                idx_start = 0
                idx_end = Nx - 1
                if a > 0:
                    bc_left = U_left
                    bc_right = None
                else:
                    bc_left = None
                    bc_right = U_right
            else:
                if a > 0:
                    x_min = -L_extra
                    x_max = x1
                else:
                    x_min = x0
                    x_max = x1 + L_extra
                Nx = int(round((x_max - x_min) / dx)) + 1
                x = np.linspace(x_min, x_max, Nx)
                idx_start = int(round((0 - x_min) / dx))
                idx_end = int(round((1 - x_min) / dx))
                if a > 0:
                    bc_left = lambda t, x0=x_min: U0(x0)
                    bc_right = None
                else:
                    bc_right = lambda t, x1=x_max: U0(x1)
                    bc_left = None
            U_sol = np.zeros((Nt + 1, idx_end - idx_start + 1))
            U_old = U0(x)
            U_sol[0, :] = U_old[idx_start:idx_end + 1]
            t_curr = t0
            for j in range(Nt):
                t_next = t_curr + dt
                U_new = np.zeros(Nx)

                if scheme == 'explicit':
                    if a > 0:
                        if bc_left is not None:
                            U_new[0] = bc_left(t_next)
                        for i in range(1, Nx):
                            U_new[i] = (U_old[i]
                                        - (a * dt / dx) * (U_old[i] - U_old[i-1])
                                        + dt * f(x[i], t_curr))
                    else:
                        if bc_right is not None:
                            U_new[Nx-1] = bc_right(t_next)
                        for i in range(0, Nx-1):
                            U_new[i] = (U_old[i]
                                        - (a * dt / dx) * (U_old[i+1] - U_old[i])
                                        + dt * f(x[i], t_curr))

                else:
                    if a > 0:
                        if bc_left is None:
                            raise ValueError("Need left BC for a>0 implicit")
                        U_new[0] = bc_left(t_next)
                        alpha = -a * dt / dx
                        beta = 1 + a * dt / dx
                        rhs = U_old + dt * f(x, t_next)
                        for i in range(1, Nx):
                            U_new[i] = (rhs[i] - alpha * U_new[i-1]) / beta
                    else:
                        if bc_right is None:
                            raise ValueError("Need right BC for a<0 implicit")
                        U_new[Nx-1] = bc_right(t_next)
                        beta = 1 - a * dt / dx
                        gamma = a * dt / dx
                        rhs = U_old + dt * f(x, t_next)
                        for i in range(Nx-2, -1, -1):
                            U_new[i] = (rhs[i] - gamma * U_new[i+1]) / beta
                U_old = U_new
                t_curr = t_next
                U_sol[j+1, :] = U_new[idx_start:idx_end + 1]
            X, T = np.meshgrid(x[idx_start:idx_end + 1], np.linspace(t0, t1, Nt+1))
            fig = plt.figure(figsize = (10, 8))
            ax = fig.add_subplot(111, projection = '3d')
            ax.plot_surface(X, T, U_sol, cmap = 'viridis', edgecolor = 'none')
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('U')
            ax.set_title(f'a = {a}, область = {region}, схема = {scheme}')
            filename = f"{output_dir}/a{a}_{region}_{scheme}.png"
            plt.savefig(filename, dpi = 150)
            plt.close(fig)
print("Результаты сохранены в папке Results_6_lab.")
