import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

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


def best_fit_c(times, x, c_grid, t_fit_max=10):
    fit_mask = times <= t_fit_max

    # errors = np.array([np.linalg.norm(x[fit_mask] - branch_path(times[fit_mask], c)) for c in c_grid])
    errors = np.array(
        [
            np.sqrt(np.mean((x[fit_mask] - branch_path(times[fit_mask], c)) ** 2))
            for c in c_grid
        ]
    )

    idx = np.argmin(errors)
    return c_grid[idx], errors[idx]


def optimal_c(times, x, c0, t_fit_max=10):
    fit_mask = times <= t_fit_max
    t_fit = times[fit_mask]
    x_fit = x[fit_mask]

    # fun = lambda c: np.sum((x_fit - branch_path(t_fit, c)) ** 2)
    fun = lambda c: np.mean((x_fit - branch_path(t_fit, c)) ** 2)

    result = optimize.minimize_scalar(
        fun, bounds=(times[0], t_fit_max), method="bounded"
    )
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

    rng = np.random.default_rng(42)
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

    c_grid = np.linspace(0, 5, 201)

    Y = analytical_solutions(times, c_grid=c_grid)

    step = 20
    for i, c in enumerate(c_grid[::step]):
        # ax.plot(times, Y[i], linestyle="--", linewidth=1, alpha=0.5, color="black")
        idx = i * step
        ax.plot(times, Y[idx], linestyle="--", linewidth=1, alpha=0.5, color="black")

    # select the smallest eps
    sample_path = X[-1, :]

    c0 = 1
    c_star_scipy, err_star_scipy = optimal_c(times, sample_path, c0, t_fit_max=10)
    print(f"best c scipy ≈ {c_star_scipy}, error scipy= {err_star_scipy}")

    c_star, err_star = best_fit_c(times, sample_path, c_grid, t_fit_max=10)
    print(f"best c ≈ {c_star}, error = {err_star}")

    ax.set_title("SDE vanishing noise vs Peano branches")
    plt.legend()
    plt.tight_layout()
    plt.show()

    delta = 1e-3

    tau = times[np.argmax(X[-1] > delta)]
    print(f"tau ≈ {tau}")
