import matplotlib.pyplot as plt
import numpy as np

from sde.numerical_schemes import explicit_euler_1d


def waiting_time_solution(b, times, tau, eps):
    """Construct a solution that stays at 0 until tau, then follows the ODE."""

    X = np.zeros(len(times))
    idx_tau = np.searchsorted(times, tau)

    for i in range(idx_tau - 1):
        dt = times[i + 1] - times[i]
        X[i + 1] = X[i] + b(X[i], times[i]) * dt

    if idx_tau < len(times):
        X[idx_tau] = eps
        for i in range(idx_tau, len(times) - 1):
            dt = times[i + 1] - times[i]
            X[i + 1] = X[i] + b(X[i], times[i]) * dt

    return X


def analytical_solutions_plot(times):
    constants = np.linspace(0, 10, 10)
    y_c = np.zeros((len(constants), len(times)))

    for i in range(len(constants)):
        c = constants[i]
        idx_c = np.searchsorted(times, c)
        y_c[i, :idx_c] = 0
        y_c[i, idx_c:] = -((times[idx_c:] - c) ** 3)

    return y_c

    def analytical_solution_plot_2(times):
        constants = np.linspace(0, 10, 10)

        # reshape per broadcasting
        T = times[None, :]  # shape (1, Nt)
        C = constants[:, None]  # shape (Nc, 1)

        y_c = -(np.maximum(T - C, 0) ** 3)

        return y_c


if __name__ == "__main__":
    t0, t1, n = 0.0, 100.0, 1000
    times = np.linspace(t0, t1, n)
    x0 = 0.0
    delta = 1e-12
    M_pow = 8

    # Non-Lipschitz drift: multiple solutions can emerge from x=0.
    b = lambda x, t: -3 * abs(x) ** (2 / 3)

    perturbs = [delta * 10**i for i in range(M_pow)]
    signs = [1.0, -1.0]

    X_all_list = []
    for s in signs:
        for p in perturbs:
            x0p = x0 + s * p
            X = explicit_euler_1d(b=b, times=times, x0=x0p)
            X_all_list.append(X)

    X_all = np.vstack(X_all_list)

    taus = np.linspace(0.0, 40.0, 7)
    eps = 1e-10
    waiting_family = [waiting_time_solution(b, times, tau=tau, eps=eps) for tau in taus]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for row in X_all:
        ax1.plot(times, row, alpha=0.7)
    ax1.set_title("Multiple deterministic solutions from tiny initial perturbations")
    ax1.set_ylabel("x(t)")

    for sol, tau in zip(waiting_family, taus):
        ax2.plot(times, sol, label=f"tau={tau:.1f}")
    ax2.set_title("Waiting-time family: same ODE, different departure times")
    ax2.set_xlabel("t")
    ax2.set_ylabel("x(t)")
    ax2.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.show()

    Y = analytical_solutions_plot(times)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(Y.shape[0]):
        ax.plot(
            times,
            Y[i, :],
            color="black",
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
            label="Analytical solutions" if i == 0 else None,
            zorder=3,
        )

    for sol, tau in zip(waiting_family, taus):
        ax.plot(
            times,
            sol,
            color="tab:blue",
            linestyle="-",
            linewidth=1.5,
            alpha=0.7,
            label=f"Numerical, tau={tau:.1f}",
            zorder=2,
        )

    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()
