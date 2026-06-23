import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama_1d

EXPERIMENT = "abs(x)_2_3"


def make_c_grid(t_fit_max=5, mode="wide"):
    if mode == "wide":
        return np.linspace(0, t_fit_max, 501)

    if mode == "refined":
        return np.linspace(0, 0.1, 201)
        # return np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.5])


def get_figures_dir(experiment: str, group: str):
    root = Path(__file__).resolve().parents[3]
    fig_dir = root / "results" / "figures" / "selection_principle" / experiment / group
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def get_data_dir(experiment: str):
    root = Path(__file__).resolve().parents[3]
    data_dir = root / "results" / "data" / "selection_principle" / experiment
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def save_experiment_data(filename, **data):
    path = get_data_dir(EXPERIMENT) / filename
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved data in {path}")


def load_experiment_data(filename):
    path = get_data_dir(EXPERIMENT) / filename
    with path.open("rb") as f:
        data = pickle.load(f)
    print(f"Loaded data from {path}")
    return data


def b(x, t):
    return 3.0 * np.abs(x) ** (2 / 3)


def branch_path(times, c):
    y = np.zeros_like(times)
    mask = times >= c
    y[mask] = (times[mask] - c) ** 3
    return y


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


def simulate_mc(times, eps, M, rng, x0=0.0, t_fit_max=5, c0=1):
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

        c_values[k], errors[k] = optimal_c(
            times=times,
            x=X[k],
            c0=c0,
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
    c0=1,
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
            rng=rng,
            t_fit_max=t_fit_max,
            c0=c0,
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
    dt_values,
    eps=1e-8,
    t0=0,
    t1=10,
    M=100,
    t_fit_max=5,
    c_grid_mode="wide",
    seed=42,
    c0=1,
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

    for dt in dt_values:
        n_steps = int(round((t1 - t0) / dt))
        if not np.isclose(n_steps * dt, t1 - t0):
            raise ValueError(
                f"dt={dt} does not divide the interval [{t0}, {t1}] into an integer number of steps"
            )
        times = np.linspace(t0, t1, n_steps + 1)
        c_grid = make_c_grid(t_fit_max=t_fit_max, mode=c_grid_mode)

        X, c_values, errors = simulate_mc(
            times=times,
            eps=eps,
            M=M,
            rng=rng,
            t_fit_max=t_fit_max,
            c0=c0,
        )

        results[dt] = {
            "dt": dt,
            "n_steps": n_steps,
            "X": X,
            "c_values": c_values,
            "errors": errors,
            "mean_c": np.mean(c_values),
            "std_c": np.std(c_values),
            "median_c": np.median(c_values),
        }

        print(
            f"n_steps={n_steps}, n_points={len(times)}, dt={dt:.2e} | "
            f"mean c={np.mean(c_values):.5g}, "
            f"std c={np.std(c_values):.5g}, "
            f"median c={np.median(c_values):.5g}"
        )

    return results


def plot_cdf_c(results, title, group, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    c_list = [res["c_values"] for res in results.values()]
    c_array = np.sort(np.array(c_list), axis=1)
    eps_vals = list(results.keys())

    total_obs = c_array.shape[1]
    for i in range(c_array.shape[0]):
        ax.plot(
            c_array[i, :],
            np.arange(1, total_obs + 1) / total_obs,
            label=rf"$\varepsilon = {eps_vals[i]:.0e}$",
        )

    plt.tight_layout()
    plt.legend()
    fig_dir = get_figures_dir(experiment=EXPERIMENT, group=group)
    plt.savefig(fig_dir / filename, dpi=300)
    plt.show()
    plt.close(fig)


def plot_c_histograms_eps(results, title, group, filename, bins=40):
    fig, ax = plt.subplots(figsize=(9, 5))

    for key, res in results.items():
        ax.hist(
            res["c_values"],
            bins=bins,
            alpha=0.45,
            label=f"eps = {key}",
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


def plot_c_histograms_grid(results, title, group, filename, bins=40):
    fig, ax = plt.subplots(figsize=(9, 5))

    for key, res in results.items():
        ax.hist(
            res["c_values"],
            bins=bins,
            alpha=0.45,
            label=f"dt = {key}",
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
    dt_values = np.array(list(results.keys()))
    mean_c = np.array([results[dt]["mean_c"] for dt in dt_values])
    std_c = np.array([results[dt]["std_c"] for dt in dt_values])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        dt_values, mean_c, yerr=std_c, marker="o", capsize=4, label="Selection of c"
    )

    ax.plot(
        dt_values - dt_values[-1],
        10 * dt_values,
        color="gray",
        linestyle="dashed",
        label="Reference slope 1",
    )
    # ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$c_\star$")
    ax.set_title(r"Dependence of selected $c_\star$ on time step")
    plt.legend(loc="upper left")
    plt.tight_layout()
    fig_dir = get_figures_dir(experiment=EXPERIMENT, group=group)
    plt.savefig(fig_dir / "c_variability.png", dpi=300)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # C_GRID_MODE = "wide"
    C_GRID_MODE = "refined"
    RECOMPUTE = False

    output_suffix = C_GRID_MODE

    eps_values = [1e-2, 1e-4, 1e-6, 1e-8]
    # eps_values = [1e-2, 1e-4]
    eps_n = 50000
    eps_M = 200
    t_fit_max = 5
    eps_cache = f"eps_results_{output_suffix}_n_{eps_n}_M_{eps_M}_tfit_{t_fit_max}.pkl"
    c0 = 0.05

    if RECOMPUTE or not (get_data_dir(EXPERIMENT) / eps_cache).exists():
        times, c_grid, eps_results = experiment_eps(
            eps_values=eps_values,
            n=eps_n,
            M=eps_M,
            t_fit_max=t_fit_max,
            c_grid_mode=C_GRID_MODE,
            c0=c0,
        )
        save_experiment_data(
            eps_cache,
            times=times,
            c_grid=c_grid,
            results=eps_results,
            eps_values=eps_values,
            c_grid_mode=C_GRID_MODE,
        )
    else:
        eps_data = load_experiment_data(eps_cache)
        times = eps_data["times"]
        c_grid = eps_data["c_grid"]
        eps_results = eps_data["results"]

    # plot_c_histograms_eps(
    #    eps_results,
    #    r"Distribution of $c_\star$ for different $\varepsilon$",
    #    group=f"eps_{output_suffix}",
    #    filename="hist.png",
    # )

    plot_cdf_c(
        eps_results,
        r"CDF of $c_\star$ for different $\varepsilon$",
        group=f"eps_{output_suffix}",
        filename="hist.png",
    )

    plot_variability_eps(
        eps_results,
        group=f"eps_{output_suffix}",
    )

    dt_values = [0.005, 0.002, 0.001, 0.0005, 0.0002]
    # dt_values = [0.005, 0.002, 0.001]
    grid_M = 200
    grid_eps = 1e-8
    grid_cache = f"grid_results_{output_suffix}_eps_{grid_eps:.0e}_M_{grid_M}_tfit_{t_fit_max}.pkl"

    if RECOMPUTE or not (get_data_dir(EXPERIMENT) / grid_cache).exists():
        grid_results = experiment_grid(
            dt_values=dt_values,
            eps=grid_eps,
            M=grid_M,
            t_fit_max=t_fit_max,
            c_grid_mode=C_GRID_MODE,
            c0=c0,
        )
        save_experiment_data(
            grid_cache,
            results=grid_results,
            dt_values=dt_values,
            eps=grid_eps,
            c_grid_mode=C_GRID_MODE,
        )
    else:
        grid_data = load_experiment_data(grid_cache)
        grid_results = grid_data["results"]

    plot_c_histograms_grid(
        grid_results,
        r"Distribution of $c_\star$ for different grid sizes",
        group=f"grid_{output_suffix}",
        filename="hist.png",
    )

    plot_variability_grid(
        grid_results,
        group=f"grid_{output_suffix}",
    )
