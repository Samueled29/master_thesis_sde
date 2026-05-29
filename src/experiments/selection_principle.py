import matplotlib.pyplot as plt
import numpy as np

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama_1d


def b(x, t):
    return -3 * abs(x) ** (2 / 3)


def analytical_solutions(times):
    constants = np.linspace(-10, 10, 10)
    # ensure 0 is included among the constants
    if not np.any(np.isclose(constants, 0.0)):
        constants = np.sort(np.append(constants, 0.0))
    y_c = np.zeros((len(constants), len(times)))

    for i in range(len(constants)):
        c = constants[i]
        idx_c = np.searchsorted(times, c)
        y_c[i, :idx_c] = 0
        y_c[i, idx_c:] = -((times[idx_c:] - c) ** 3)

    return y_c, constants


if __name__ == "__main__":
    t0 = 0
    t1 = 50
    n = 1000
    times = np.linspace(t0, t1, n)
    x0 = 0

    rng = np.random.default_rng(42)

    M = 20
    eps_list = [500, 100, 50, 10, 0.1, 1e-2, 1e-5, 1e-8]
    X = np.zeros((len(eps_list), len(times)))

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, eps in enumerate(eps_list):
        X_loc = np.zeros((M, len(times)))
        sigma = lambda x, t: eps

        for j in range(M):
            W = brownian_motion(times)
            X_loc[j, :] = euler_maruyama_1d(
                b=b, sigma=sigma, times=times, x0=x0, W=W, rng=None
            )

        X[i, :] = np.mean(X_loc, axis=0)
        ax.plot(times, X[i, :], label=f"$\\epsilon={eps}$")

    Y, c = analytical_solutions(times)

    for i in range(Y.shape[0]):
        ax.plot(
            times,
            Y[i, :],
            color=("black" if c[i] != 0 else "red"),
            linestyle="--",
            linewidth=1,
            alpha=0.9,
            label=f"c={c[i]}",
            zorder=3,
        )

    ax.legend()
    plt.tight_layout()
    plt.show()
