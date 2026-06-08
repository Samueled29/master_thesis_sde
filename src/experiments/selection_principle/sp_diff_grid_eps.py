from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama_1d

EXPERIMENT = "abs(x)_2_3"


def make_c_grid(t_fit_max=5, mode="wide"):
    if mode == "wide":
        return np.linspace(0, t_fit_max, 501)

    if mode == "refined":
        return np.linspace(0, 0.05, 201)


def get_figures_dir(experiment: str, group: str):
    root = Path(__file__).resolve().parents[3]
    fig_dir = root / "results" / "figures" / "selection_principle" / experiment / group
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def b(x, t):
    return 3.0 * np.abs(x) ** (2 / 3)


def branch_path(times, c):
    y = np.zeros_like(times)
    mask = times >= c
    y[mask] = (times[mask] - c) ** 3
    return y


def best_fit_c(times, x, c_grid, t_fit_max=5):
    """Compare numerical solution x to analytical branches for different values of c
    that is (y-c)^3, then compute the errors and return the best fitting branch, and its error
    to have a better error computation, errors are only computed up to a time t_fit_max to avoid computing error
    after the interesting points."""
    mask = times <= t_fit_max

    errors = np.array(
        [np.linalg.norm(x[mask] - branch_path(times[mask], c)) for c in c_grid]
    )

    idx = np.argmin(errors)
    return c_grid[idx], errors[idx]


def simulate_mc(times, eps, M, c_grid, rng, x0=0.0, t_fit_max=5):
    """Run simulation of Euler Maruyama for M BM realizations. for every realization, compute the best branch
    using best_fit_c.
    Return numerical solutions X, best branches and errors for every trajectory."""

    X = np.zeros((M, len(times)))
    c_values = np.zeros(M)
    errors = np.zeros(M)

    sigma = lambda x, t: eps

    for k in range(M):
        W = brownian_motion(times, rng=rng)

        X[k] = euler_maruyama_1d(
            b=b,
            sigma=sigma,
            times=times,
            x0=x0,
            W=W,
            rng=None,
        )

        c_values[k], errors[k] = best_fit_c(
            times=times,
            x=X[k],
            c_grid=c_grid,
            t_fit_max=t_fit_max,
        )

    return X, c_values, errors


def experiment_eps(
    eps_values,
    t0=0,
    t1=10,
    n=50000,
    M=100,
    t_fit_max=5,
    c_grid_mode="wide",
    seed=42,
):
    """
    Run Monte Carlo simulations for different values of noise intensity, eps.
    Return a dictionary with noise intensity eps as keys. for every value of eps there is a dictionary
    containing:
    X
    best values of c for every trajectory
    errors
    mean value of c
    std deviation
    median"""
    rng = np.random.default_rng(seed)
    times = np.linspace(t0, t1, n)
    c_grid = make_c_grid(t_fit_max=t_fit_max, mode=c_grid_mode)

    results = {}

    for eps in eps_values:
        X, c_values, errors = simulate_mc(
            times=times,
            eps=eps,
            M=M,
            c_grid=c_grid,
            rng=rng,
            t_fit_max=t_fit_max,
        )

        results[eps] = {
            "X": X,
            "c_values": c_values,
            "errors": errors,
            "mean_c": np.mean(c_values),
            "std_c": np.std(c_values),
            "median_c": np.median(c_values),
        }

        print(
            f"eps={eps:.1e} | "
            f"mean c={np.mean(c_values):.5g}, "
            f"std c={np.std(c_values):.5g}, "
            f"median c={np.median(c_values):.5g}"
        )

    return times, c_grid, results


def experiment_grid(
    n_values,
    eps=1e-8,
    t0=0,
    t1=10,
    M=100,
    t_fit_max=5,
    c_grid_mode="wide",
    seed=42,
):
    """Run MC simulations for a fixed value of noise intensity, for different values of grid size.
    return a dictionary with number of grid points as keys.
    for every item in dictionary, there is a dictionary with:
    dt,
    X numerical simulation,
    values of best c for every trajectory
    errors,
    statistics
    """
    rng = np.random.default_rng(seed)
    results = {}

    for n in n_values:
        times = np.linspace(t0, t1, n)
        c_grid = make_c_grid(t_fit_max=t_fit_max, mode=c_grid_mode)

        X, c_values, errors = simulate_mc(
            times=times,
            eps=eps,
            M=M,
            c_grid=c_grid,
            rng=rng,
            t_fit_max=t_fit_max,
        )

        dt = times[1] - times[0]

        results[n] = {
            "dt": dt,
            "X": X,
            "c_values": c_values,
            "errors": errors,
            "mean_c": np.mean(c_values),
            "std_c": np.std(c_values),
            "median_c": np.median(c_values),
        }

        print(
            f"n={n}, dt={dt:.2e} | "
            f"mean c={np.mean(c_values):.5g}, "
            f"std c={np.std(c_values):.5g}, "
            f"median c={np.median(c_values):.5g}"
        )

    return results


def plot_c_histograms(results, title, group, filename, bins=40):
    fig, ax = plt.subplots(figsize=(9, 5))

    for key, res in results.items():
        ax.hist(
            res["c_values"],
            bins=bins,
            alpha=0.45,
            label=f"{key}",
        )

    ax.set_title(title)
    ax.set_xlabel(r"$c_\star$")
    ax.set_ylabel("count")
    ax.legend()
    plt.tight_layout()

    fig_dir = get_figures_dir(experiment=EXPERIMENT, group=group)
    plt.savefig(fig_dir / filename, dpi=300)
    plt.show()
    plt.close(fig)


def plot_variability_eps(results, group="eps"):
    """Plot variability of the selected constants depending on the noise intensity."""
    eps_values = np.array(list(results.keys()))
    mean_c = np.array([results[e]["mean_c"] for e in eps_values])
    std_c = np.array([results[e]["std_c"] for e in eps_values])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(eps_values, mean_c, yerr=std_c, marker="o", capsize=4)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$c_\star$")
    ax.set_title(r"Dependence of selected $c_\star$ on noise level")
    plt.tight_layout()
    fig_dir = get_figures_dir(experiment=EXPERIMENT, group=group)
    plt.savefig(fig_dir / "c_variability.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_variability_grid(results, group="grid"):
    """Plot variability of c for different grid sizes, fixed noise intensity."""
    n_values = np.array(list(results.keys()))
    dt_values = np.array([results[n]["dt"] for n in n_values])
    mean_c = np.array([results[n]["mean_c"] for n in n_values])
    std_c = np.array([results[n]["std_c"] for n in n_values])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(dt_values, mean_c, yerr=std_c, marker="o", capsize=4)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$c_\star$")
    ax.set_title(r"Dependence of selected $c_\star$ on time step")
    plt.tight_layout()
    fig_dir = get_figures_dir(experiment=EXPERIMENT, group=group)
    plt.savefig(fig_dir / "c_variability.png", dpi=300)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # C_GRID_MODE = "wide"
    C_GRID_MODE = "refined"

    output_suffix = C_GRID_MODE

    eps_values = [1e-2, 1e-4, 1e-6, 1e-8]

    times, c_grid, eps_results = experiment_eps(
        eps_values=eps_values,
        n=50000,
        M=100,
        t_fit_max=5,
        c_grid_mode=C_GRID_MODE,
    )

    plot_c_histograms(
        eps_results,
        r"Distribution of $c_\star$ for different $\varepsilon$",
        group=f"eps_{output_suffix}",
        filename="hist.png",
    )

    plot_variability_eps(
        eps_results,
        group=f"eps_{output_suffix}",
    )

    n_values = [2000, 5000, 10000, 20000, 50000]

    grid_results = experiment_grid(
        n_values=n_values,
        eps=1e-8,
        M=100,
        t_fit_max=5,
        c_grid_mode=C_GRID_MODE,
    )

    plot_c_histograms(
        grid_results,
        r"Distribution of $c_\star$ for different grid sizes",
        group=f"grid_{output_suffix}",
        filename="hist.png",
    )

    plot_variability_grid(
        grid_results,
        group=f"grid_{output_suffix}",
    )
