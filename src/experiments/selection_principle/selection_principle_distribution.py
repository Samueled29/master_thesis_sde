import matplotlib.pyplot as plt
import numpy as np

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama_1d


def b(x, t):
    return 3 * np.abs(x) ** (2 / 3)


def branch_path(times, c):
    y = np.zeros_like(times)
    mask = times >= c
    y[mask] = (times[mask] - c) ** 3
    return y


def analytical_solutions(times, c_grid):
    Y = np.zeros((len(c_grid), len(times)))
    for i, c in enumerate(c_grid):
        Y[i] = branch_path(times, c)

    return Y


def best_fit_c(times, x, c_grid, t_fit_max=5):

    mask = times <= t_fit_max

    errors = np.array(
        [np.linalg.norm(x[mask] - branch_path(times[mask], c)) for c in c_grid]
    )

    idx = np.argmin(errors)

    return c_grid[idx], errors[idx]


if __name__ == "__main__":
    t0 = 0
    t1 = 10
    n = 50000
    times = np.linspace(t0, t1, n)
    x0 = 0

    rng = np.random.default_rng(42)

    M = 100
    eps = 1e-6
    c_values = np.zeros(M)

    c_grid = np.linspace(0, 0.5, 501)

    n_t = len(times)
    X = np.zeros((M, n_t))

    sigma = lambda x, t: eps

    for k in range(M):
        W = brownian_motion(times, rng=rng)

        X[k] = euler_maruyama_1d(b=b, sigma=sigma, times=times, x0=x0, W=W, rng=None)
        c_star, err_star = best_fit_c(times, X[k], c_grid, t_fit_max=5)
        c_values[k] = c_star

    fig, ax = plt.subplots(figsize=(12, 8))

    for k in range(min(M, 20)):
        ax.plot(times, X[k], alpha=0.5)

    ax.set_title("Vanishing-noise trajectories")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(c_values, bins=20)

    ax.set_title(r"Distribution of selected $c_\star$")
    ax.set_xlabel(r"$c_\star$")
    plt.show()
