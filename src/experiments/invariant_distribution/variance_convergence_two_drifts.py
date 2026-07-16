from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama_1d


@dataclass(frozen=True)
class DriftConfig:
    name: str
    label: str
    drift: Callable[[np.ndarray, float], np.ndarray]
    true_variance: Callable[[float], float]


def get_figures_dir():
    root = Path(__file__).resolve().parents[3]
    fig_dir = root / "results" / "figures" / "variance_analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def holder_drift(x, t):
    return -3.0 * np.sign(x) * np.abs(x) ** (2 / 3)


def holder_true_variance(eps):
    return (5 * eps**2 / 18) ** (6 / 5) * gamma(9 / 5) / gamma(3 / 5)


LAMBDA = 3.0


def regular_drift(x, t):
    return -LAMBDA * x


def regular_true_variance(eps):
    return eps**2 / (2 * LAMBDA)


DRIFTS = [
    DriftConfig(
        name="regular",
        label="Regular drift",
        drift=regular_drift,
        true_variance=regular_true_variance,
    ),
    DriftConfig(
        name="holder",
        label="Holder drift",
        drift=holder_drift,
        true_variance=holder_true_variance,
    ),
]


def estimate_invariant_variance(
    drift_config,
    dt,
    eps,
    n_replications,
    t0=0.0,
    t_burn=2000.0,
    t1=32000.0,
    x0=0.0,
    seed0=2000,
):
    # Use independent long trajectories: burn-in controls invariant-regime bias,
    # independent replications give meaningful error bars for the variance bias.
    sigma = lambda x, t: eps
    times = np.arange(t0, t1 + dt, dt)
    invariant_mask = times >= t_burn

    estimates = np.zeros(n_replications)

    for r in range(n_replications):
        rng = np.random.default_rng(seed0 + r)
        W = brownian_motion(times=times, rng=rng)
        X = euler_maruyama_1d(
            b=drift_config.drift,
            sigma=sigma,
            times=times,
            x0=x0,
            W=W,
            rng=None,
        )
        estimates[r] = np.var(X[invariant_mask])

    return estimates


def run_variance_convergence(
    drift_config,
    dt_values,
    eps=1.0,
    n_replications=50,
    fit_dt_values=None,
):
    var_true = drift_config.true_variance(eps)
    bias_abs = []
    bias_std_error = []

    print(f"\n{drift_config.label}")
    print(f"Analytical variance: {var_true:.8e}\n")

    for dt in dt_values:
        estimates = estimate_invariant_variance(
            drift_config=drift_config,
            dt=dt,
            eps=eps,
            n_replications=n_replications,
        )
        bias = estimates - var_true
        bias_mean = np.mean(bias)

        bias_abs.append(abs(bias_mean))
        bias_std_error.append(np.std(bias, ddof=1) / np.sqrt(n_replications))

        print(f"dt = {dt:<6} | Bias: {bias_mean:+.6e}")

    dt_values = np.array(dt_values, dtype=float)
    bias_abs = np.array(bias_abs)
    bias_std_error = np.array(bias_std_error)

    if fit_dt_values is None:
        fit_dt_values = dt_values[-3:]

    fit_dt_values = np.array(fit_dt_values, dtype=float)
    fit_mask = np.isin(dt_values, fit_dt_values)
    if np.count_nonzero(fit_mask) != len(fit_dt_values):
        raise ValueError("Some fit_dt_values are not present in dt_values.")

    rate, intercept = np.polyfit(
        np.log(dt_values[fit_mask]),
        np.log(bias_abs[fit_mask]),
        1,
    )

    fit_points_text = ", ".join(f"{dt:g}" for dt in dt_values[fit_mask])
    print(f"Fit points kept for asymptotic regime: {fit_points_text}")
    print(f"Estimated weak rate: {rate:.3f}")

    return {
        "dt_values": dt_values,
        "bias_abs": bias_abs,
        "bias_std_error": bias_std_error,
        "rate": rate,
        "intercept": intercept,
        "fit_dt_values": dt_values[fit_mask],
        "var_true": var_true,
    }


def plot_convergence(result, drift_config, eps, n_replications):
    dt_values = result["dt_values"]
    bias_abs = result["bias_abs"]
    bias_std_error = result["bias_std_error"]
    rate = result["rate"]
    intercept = result["intercept"]

    fit_line = np.exp(intercept) * dt_values**rate
    ref_line = (bias_abs[1] / dt_values[1]) * dt_values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        dt_values,
        bias_abs,
        yerr=bias_std_error,
        marker="o",
        capsize=4,
        linestyle="None",
        label="EM",
    )
    ax.loglog(dt_values, fit_line, "--", label=f"Fit slope (order)={rate:.2f}")
    ax.loglog(dt_values, ref_line, ":", color="gray", label="Reference order 1.0")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$|\operatorname{Var}_{EM}-\operatorname{Var}_{true}|$")
    ax.set_title(drift_config.label)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()

    fig_dir = get_figures_dir()
    filename = (
        f"variance_convergence_{drift_config.name}_"
        f"eps_{eps:g}_R_{n_replications}.png"
    )
    plt.savefig(fig_dir / filename, dpi=300)
    print(f"Saved plot in {fig_dir / filename}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    dt_values = [0.08, 0.04, 0.02, 0.01, 0.005]
    fit_dt_values = [0.02, 0.01, 0.005]
    eps = 1.0
    n_replications = 50

    print(
        "Using asymptotic fit points: "
        + ", ".join(f"{dt:g}" for dt in fit_dt_values)
    )

    for drift_config in DRIFTS:
        result = run_variance_convergence(
            drift_config=drift_config,
            dt_values=dt_values,
            eps=eps,
            n_replications=n_replications,
            fit_dt_values=fit_dt_values,
        )
        plot_convergence(
            result,
            drift_config,
            eps=eps,
            n_replications=n_replications,
        )