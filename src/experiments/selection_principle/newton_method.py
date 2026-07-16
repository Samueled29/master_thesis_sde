from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama_1d


def get_figures_dir():
    root = Path(__file__).resolve().parents[3]
    fig_dir = root / "results" / "figures" / "selection_principle" / "pathwise_solution"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


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

def optimal_c(times, x, c0, t_fit_max=10):
    fit_mask = times <= t_fit_max
    t_fit = times[fit_mask]
    x_fit = x[fit_mask]

    # fun = lambda c: np.sum((x_fit - branch_path(t_fit, c)) ** 2)
    fun = lambda c: np.mean((x_fit - branch_path(t_fit, c)) ** 2)

    #result = optimize.minimize_scalar(fun, bounds=(times[0], t_fit_max), method="bounded")
    result = optimize.minimize_scalar(fun, method = "brent")
    #result = optimize.minimize(fun, c0,  method = "Newton-CG")
    c_star = result.x
    err_star = result.fun

    return c_star, err_star


if __name__ == "__main__":
    t0 = 0
    t1 = 10
    # n = 50000
    dt = 2e-5
    times = np.arange(t0, t1 + dt, dt)
    x0 = 0

    rng = np.random.default_rng(789)
    W = brownian_motion(times, rng=rng)

    eps_list = np.logspace(1, -6, 8)

    n_eps = len(eps_list)
    n_t = len(times)

    X = np.zeros((n_eps, n_t))

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, eps in enumerate(eps_list):
        sigma = lambda x, t: eps

        X[i, :] = euler_maruyama_1d(b=b, sigma=sigma, times=times, x0=x0, W=W, rng=None)

        ax.plot(times, X[i, :], linewidth=2, label=rf"$\varepsilon={eps}$")

    c_grid = np.linspace(-0.5, 2, 5)
    if 0 not in c_grid:
        idx = np.searchsorted(c_grid, 0)
        c_grid = np.insert(c_grid, idx, 0)

    Y = analytical_solutions(times, c_grid=c_grid)

    for i in range(len(Y)):
        ax.plot(times, Y[i], linestyle="--", linewidth=1, alpha=0.5, color="black")

    # select the smallest eps
    sample_path = X[-1, :]

    c0 = 1.0
    c_star_scipy, err_star_scipy = optimal_c(times, sample_path, c0, t_fit_max=10)
    print(f"best c scipy ≈ {c_star_scipy}, error scipy= {err_star_scipy}")

    ax.set_title("Zero-noise selection principle")
    plt.ylabel(r'X_t($\omega$)')
    plt.xlabel('t')
    plt.grid(True, which="both")
    plt.legend(loc = "upper left")
    plt.tight_layout()
    fig_dir = get_figures_dir()
    plt.savefig(fig_dir / "pathwise_selection_solution.png", dpi=300)
    plt.show()

    delta = 1e-3

    tau = times[np.argmax(X[-1] > delta)]
    print(f"tau ≈ {tau}")
