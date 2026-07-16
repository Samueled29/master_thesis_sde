import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama_1d


def get_figures_dir(group: str):
    root = Path(__file__).resolve().parents[3]
    fig_dir = root / "results" / "figures" / "selection_principle" / group
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def get_data_dir():
    root = Path(__file__).resolve().parents[3]
    data_dir = root / "results" / "data" / "selection_principle"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def save_experiment_data(filename, **data):
    path = get_data_dir() / filename
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved data in {path}")


def load_experiment_data(filename):
    path = get_data_dir() / filename
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


def fit_branch_shift(times, x, t_fit_max=10, fit_bounds=None):
    """Fit the analytical branch (t - c)^3 by optimizing over c."""
    fit_mask = times <= t_fit_max
    t_fit = times[fit_mask]
    x_fit = x[fit_mask]

    if fit_bounds is None:
        fit_bounds = (times[0], t_fit_max)

    def mean_squared_error(c):
        return np.mean((x_fit - branch_path(t_fit, c)) ** 2)

    result = optimize.minimize_scalar(
        mean_squared_error, bounds=fit_bounds, method="bounded"
    )
    if not result.success:
        raise RuntimeError(f"Could not fit branch shift: {result.message}")

    return result.x, result.fun


def summarize_simulation(X, c_values, errors, **metadata):
    return {
        **metadata,
        "X": X,
        "c_values": c_values,
        "errors": errors,
        "mean_c": np.mean(c_values),
        "std_c": np.std(c_values),
        "median_c": np.median(c_values),
    }


def simulate_mc(times, eps, M, rng, x0=0.0, t_fit_max=5, fit_bounds=None):
    """Run M Euler-Maruyama simulations and fit one branch shift per path."""
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
        c_values[k], errors[k] = fit_branch_shift(
            times=times,
            x=X[k],
            t_fit_max=t_fit_max,
            fit_bounds=fit_bounds,
        )

    return X, c_values, errors


def experiment_eps(
    eps_values,
    t0=0,
    t1=10,
    n=50000,
    M=100,
    t_fit_max=5,
    seed=42,
):
    """Run Monte Carlo simulations for different noise intensities."""
    rng = np.random.default_rng(seed)
    times = np.linspace(t0, t1, n)
    results = {}

    for eps in eps_values:
        X, c_values, errors = simulate_mc(
            times=times,
            eps=eps,
            M=M,
            rng=rng,
            t_fit_max=t_fit_max,
        )
        results[eps] = summarize_simulation(X, c_values, errors)
        print(
            f"eps={eps:.1e} | "
            f"mean c={np.mean(c_values):.5g}, "
            f"std c={np.std(c_values):.5g}, "
            f"median c={np.median(c_values):.5g}"
        )

    return times, results


def experiment_grid(
    dt_values,
    eps=1e-8,
    t0=0,
    t1=10,
    M=100,
    t_fit_max=5,
    seed=42,
    x0=0.0,
    fit_bounds=None,
):
    """Run Monte Carlo simulations with fixed noise and different time steps."""
    rng = np.random.default_rng(seed)
    results = {}

    for dt in dt_values:
        n_steps = int(round((t1 - t0) / dt))
        if not np.isclose(n_steps * dt, t1 - t0):
            raise ValueError(
                f"dt={dt} does not divide the interval [{t0}, {t1}] into an integer number of steps"
            )
        times = np.linspace(t0, t1, n_steps + 1)
        X, c_values, errors = simulate_mc(
            times=times,
            eps=eps,
            M=M,
            rng=rng,
            x0=x0,
            t_fit_max=t_fit_max,
            fit_bounds=fit_bounds,
        )
        results[dt] = summarize_simulation(
            X, c_values, errors, dt=dt, n_steps=n_steps, x0=x0
        )
        print(
            f"x0={x0:g}, n_steps={n_steps}, n_points={len(times)}, dt={dt:.2e} | "
            f"mean c={np.mean(c_values):.5g}, "
            f"std c={np.std(c_values):.5g}, "
            f"median c={np.median(c_values):.5g}"
        )

    return results


def plot_cdf_c(results, title, group, filename, label_format):
    fig, ax = plt.subplots(figsize=(9, 5))
    c_list = [res["c_values"] for res in results.values()]
    c_array = np.sort(np.array(c_list), axis=1)
    labels = list(results.keys())
    total_obs = c_array.shape[1]

    for i in range(c_array.shape[0]):
        ax.plot(
            c_array[i, :],
            np.arange(1, total_obs + 1) / total_obs,
            label=label_format(labels[i]),
        )

    ax.set_title(title)
    ax.set_xlabel(r"$c_\star$")
    ax.set_ylabel("CDF")
    ax.legend()
    plt.tight_layout()
    fig_dir = get_figures_dir(group)
    plt.savefig(fig_dir / filename, dpi=300)
    plt.show()
    plt.close(fig)


def plot_c_histograms(results, title, group, filename, label_format, bins=40):
    fig, ax = plt.subplots(figsize=(9, 5))

    for key, res in results.items():
        ax.hist(
            res["c_values"],
            bins=bins,
            alpha=0.45,
            label=label_format(key),
        )

    ax.set_title(title)
    ax.set_xlabel(r"$c_\star$")
    ax.set_ylabel("count")
    ax.legend()
    plt.tight_layout()
    fig_dir = get_figures_dir(group)
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
    fig_dir = get_figures_dir(group)
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
        10* dt_values,
        color="gray",
        linestyle="dashed",
        label="Reference slope 1",
    )
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$c_\star$")
    ax.set_title(r"Dependence of selected $c_\star$ on time step")
    plt.legend(loc="upper left")
    plt.tight_layout()
    fig_dir = get_figures_dir(group)
    plt.savefig(fig_dir / "c_variability.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_grid_initial_condition_variability(results_by_x0, group="grid_initial_conditions"):
    """Compare translated selected c values on a log-log scale."""
    reference_c = {0.0: 0.0, 1.0: -1.0}

    fig, ax = plt.subplots(figsize=(8, 5))
    reference_dt_values = None
    reference_error = None

    for x0, results in results_by_x0.items():
        dt_values = np.array(list(results.keys()))
        mean_c = np.array([results[dt]["mean_c"] for dt in dt_values])
        translated_error = np.abs(mean_c - reference_c[x0])

        order = np.argsort(dt_values)
        dt_values = dt_values[order]
        translated_error = translated_error[order]

        ax.loglog(
            dt_values,
            translated_error,
            marker="o",
            label=rf"$X_0 = {x0:g}$",
        )

        if reference_dt_values is None:
            reference_dt_values = dt_values
            reference_error = translated_error[-1]

    ax.loglog(
        reference_dt_values,
        reference_error * reference_dt_values / reference_dt_values[-1],
        color="gray",
        linestyle="dashed",
        label="Reference slope 1",
    )
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$|\mathbb{E}[c_\star] - c_{ref}|$")
    ax.set_title(r"Translated selected $c_\star$ for different initial conditions")
    ax.legend(loc="best")
    ax.grid(visible= True, axis = 'both')
    plt.tight_layout()
    fig_dir = get_figures_dir(group)
    plt.savefig(fig_dir / "c_variability_initial_conditions.png", dpi=300)
    plt.show()
    plt.close(fig)

def plot_grid_initial_condition_convergence(results_by_x0, group="grid_initial_conditions"):
    """Compare time-step scaling of the fitted branch shift for different initial data."""
    reference_c = {0.0: 0.0, 1.0: -1.0}

    fig, ax = plt.subplots(figsize=(8, 5))
    for x0, results in results_by_x0.items():
        dt_values = np.array(list(results.keys()))
        mean_c = np.array([results[dt]["mean_c"] for dt in dt_values])
        error_c = np.abs(mean_c - reference_c[x0])
        ax.loglog(dt_values, error_c, marker="o", label=rf"$X_0 = {x0:g}$")

    dt_values = np.array(list(next(iter(results_by_x0.values())).keys()))
    ax.loglog(
        dt_values,
        dt_values / dt_values[-1] * 1e-3,
        color="gray",
        linestyle="dashed",
        label="Reference slope 1",
    )
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$|\mathbb{E}[c_\star] - c_{ref}|$")
    ax.set_title(r"Time-step scaling for different initial conditions")
    ax.legend(loc="best")
    plt.tight_layout()
    fig_dir = get_figures_dir(group)
    plt.savefig(fig_dir / "c_convergence_initial_conditions.png", dpi=300)
    plt.show()
    plt.close(fig)


def grid_cache_name(fit_method, eps, M, t_fit_max, x0=0.0):
    if x0 == 0.0:
        return f"grid_results_{fit_method}_eps_{eps:.0e}_M_{M}_tfit_{t_fit_max}.pkl"
    return f"grid_results_x0_{x0:g}_{fit_method}_eps_{eps:.0e}_M_{M}_tfit_{t_fit_max}.pkl"


def load_or_run_grid(cache, recompute, dt_values, eps, M, t_fit_max, x0, fit_bounds=None):
    if recompute or not (get_data_dir() / cache).exists():
        results = experiment_grid(
            dt_values=dt_values,
            eps=eps,
            M=M,
            t_fit_max=t_fit_max,
            x0=x0,
            fit_bounds=fit_bounds,
        )
        save_experiment_data(
            cache,
            results=results,
            dt_values=dt_values,
            eps=eps,
            x0=x0,
            fit_bounds=fit_bounds,
            fit_method="scalar_fit",
        )
        return results

    data = load_experiment_data(cache)
    return data["results"]


if __name__ == "__main__":
    RECOMPUTE = False
    fit_method = "scalar_fit"

    eps_values = [1e-2, 1e-4, 1e-6, 1e-8]
    eps_n = 50000
    eps_M = 200
    t_fit_max = 5
    eps_cache = f"eps_results_{fit_method}_n_{eps_n}_M_{eps_M}_tfit_{t_fit_max}.pkl"

    if RECOMPUTE or not (get_data_dir() / eps_cache).exists():
        times, eps_results = experiment_eps(
            eps_values=eps_values,
            n=eps_n,
            M=eps_M,
            t_fit_max=t_fit_max,
        )
        save_experiment_data(
            eps_cache,
            times=times,
            results=eps_results,
            eps_values=eps_values,
            fit_method=fit_method,
        )
    else:
        eps_data = load_experiment_data(eps_cache)
        times = eps_data["times"]
        eps_results = eps_data["results"]

    plot_cdf_c(
        eps_results,
        r"CDF of $c_\star$ for different $\varepsilon$",
        group=f"eps_{fit_method}",
        filename="cdf.png",
        label_format=lambda eps: rf"$\varepsilon = {eps:.0e}$",
    )
    plot_variability_eps(
        eps_results,
        group=f"eps_{fit_method}",
    )

    dt_values = [0.005, 0.002, 0.001, 0.0005, 0.0002]
    grid_M = 200
    grid_eps = 1e-8

    grid_results_x0_0 = load_or_run_grid(
        cache=grid_cache_name(fit_method, grid_eps, grid_M, t_fit_max, x0=0.0),
        recompute=RECOMPUTE,
        dt_values=dt_values,
        eps=grid_eps,
        M=grid_M,
        t_fit_max=t_fit_max,
        x0=0.0,
    )
    grid_results_x0_1 = load_or_run_grid(
        cache=grid_cache_name(fit_method, grid_eps, grid_M, t_fit_max, x0=1.0),
        recompute=RECOMPUTE,
        dt_values=dt_values,
        eps=grid_eps,
        M=grid_M,
        t_fit_max=t_fit_max,
        x0=1.0,
        fit_bounds=(-2.0, 0.0),
    )

    plot_c_histograms(
        grid_results_x0_0,
        r"Distribution of $c_\star$ for different grid sizes, $X_0 = 0$",
        group=f"grid_{fit_method}",
        filename="hist.png",
        label_format=lambda dt: f"dt = {dt}",
    )
    plot_variability_grid(
        grid_results_x0_0,
        group=f"grid_{fit_method}",
    )
    grid_results_by_x0 = {0.0: grid_results_x0_0, 1.0: grid_results_x0_1}
    plot_grid_initial_condition_variability(
        grid_results_by_x0,
        group=f"grid_initial_conditions_{fit_method}",
    )
    plot_grid_initial_condition_convergence(
        grid_results_by_x0,
        group=f"grid_initial_conditions_{fit_method}",
    )