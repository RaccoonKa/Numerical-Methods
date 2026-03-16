import numpy as np
import csv
import os

output_dir = 'Results_7_lab'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

L = 1.0
T_end = 1.0
h = 0.01
tau = 0.0001
eps = 0.005

nx = int(L / h) + 1
nt = int(T_end / tau) + 1

x = np.linspace(0, L, nx)
t = np.linspace(0, T_end, nt)

def get_initial_u(x_arr):
    return np.where(x_arr < 0.5, 3.0, 1.0)

U_visc = np.zeros((nt, nx))
U_visc[0, :] = get_initial_u(x)

for j in range(0, nt - 1):
    U_visc[j + 1, 0] = 3.0
    for i in range(1, nx - 1):
        advection = (tau / h) * U_visc[j, i] * (U_visc[j, i] - U_visc[j, i - 1])
        diff_right = (U_visc[j, i + 1] - U_visc[j, i]) ** 2
        diff_left = (U_visc[j, i] - U_visc[j, i - 1]) ** 2
        viscosity = (eps * tau / (2 * h ** 3)) * (diff_right - diff_left)
        U_visc[j + 1, i] = U_visc[j, i] - advection - viscosity
    U_visc[j + 1, -1] = U_visc[j + 1, -2]

U_cons = np.zeros((nt, nx))
U_cons[0, :] = get_initial_u(x)

for j in range(0, nt - 1):
    U_cons[j + 1, 0] = 3.0
    for i in range(1, nx):
        flux_term = (tau / (2 * h)) * (U_cons[j, i] ** 2 - U_cons[j, i - 1] ** 2)
        U_cons[j + 1, i] = U_cons[j, i] - flux_term


def save_results(filename, data):
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['t/x'] + [round(val, 2) for val in x[::10]])
        for i in range(0, nt, 1000):
            row = [round(t[i], 2)] + [round(val, 4) for val in data[i, ::10]]
            writer.writerow(row)

save_results('viscosity_method.csv', U_visc)
save_results('conservative_method.csv', U_cons)

print(f"Результаты сохранены в папку: {os.path.abspath(output_dir)}")
