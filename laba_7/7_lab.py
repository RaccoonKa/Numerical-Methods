import numpy as np
import pandas as pd
from pathlib import Path


def initial_condition(x):
    if x >= 0.5:
        return 1.0
    else:
        return 3.0


def artificial_viscosity_scheme(a, b, c, d, h, tau):
    x = np.arange(a, b + h, h)
    t = np.arange(c, d + tau, tau)
    nx = len(x)
    nt = len(t)
    U = np.zeros((nt, nx))
    for j in range(nx):
        U[0, j] = initial_condition(x[j])

    for n in range(nt - 1):

        for j in range(1, nx - 1):

            u = U[n, j]

            U[n + 1, j] = (
                U[n, j]
                - tau * u * (U[n, j] - U[n, j - 1]) / h
                + 0.5 * tau * (U[n, j + 1] - 2 * U[n, j] + U[n, j - 1]) / h
            )

        U[n + 1, 0] = U[n + 1, 1]
        U[n + 1, -1] = U[n + 1, -2]

    return x, t, U


def conservative_scheme(a, b, c, d, h, tau):
    x = np.arange(a, b + h, h)
    t = np.arange(c, d + tau, tau)
    nx = len(x)
    nt = len(t)
    U = np.zeros((nt, nx))
    for j in range(nx):
        U[0, j] = initial_condition(x[j])

    for n in range(nt - 1):

        for j in range(1, nx - 1):

            f_plus = 0.5 * U[n, j + 1] ** 2
            f_minus = 0.5 * U[n, j - 1] ** 2
            U[n + 1, j] = (
                0.5 * (U[n, j + 1] + U[n, j - 1])
                - tau / (2 * h) * (f_plus - f_minus)
            )

        U[n + 1, 0] = U[n + 1, 1]
        U[n + 1, -1] = U[n + 1, -2]

    return x, t, U


def save_to_csv(x, t, U, filename):
    df = pd.DataFrame(U, columns = [f"x={round(val,4)}" for val in x])
    df.insert(0, "t", t)
    df.to_csv(filename, index = False)


def main():
    a = 0
    b = 1
    c = 0
    d = 1
    h = 0.02
    tau = 0.005
    results_dir = Path("Results_7_lab")
    results_dir.mkdir(exist_ok = True)
    x1, t1, U1 = artificial_viscosity_scheme(a, b, c, d, h, tau)
    save_to_csv(x1, t1, U1, results_dir / "artificial_viscosity.csv")
    x2, t2, U2 = conservative_scheme(a, b, c, d, h, tau)
    save_to_csv(x2, t2, U2, results_dir / "conservative_scheme.csv")
    print("Результаты сохранены в папке Results_7_lab")


if __name__ == "__main__":
    main()
