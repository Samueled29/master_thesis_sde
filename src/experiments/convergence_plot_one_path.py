import matplotlib.pyplot as plt
import numpy as np

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama_1d


def b(x, t):
    return -3 * abs(x) ** (2 / 3)


if __name__ == "__main__":
    t0 = 0
    t1 = 1
    h_ref = 1e-6
    n_ref = int((t1 - t0) / h_ref) + 1
    times = np.linspace(t0, t1, n_ref)
    x0 = 0

    M = 100

    h_list = [10 ** (-i) for i in range(1, 5)]
    strides = [int(round(h / h_ref)) for h in h_list]
    strides = [max(1, s) for s in strides]

    seed = 42
    rng = np.random.default_rng(seed)
    W = brownian_motion(times=times, rng=rng)

    eps = 100
    sigma = lambda x, t: np.sqrt(eps)

    X_ref = euler_maruyama_1d(b=b, sigma=sigma, times=times, x0=0, W=W)

    X_sim = []

    for stride in strides:
        times_h = times[::stride]
        W_h = W[::stride]
        X_h = euler_maruyama_1d(b=b, sigma=sigma, times=times_h, x0=0, W=W_h)
        X_sim.append(X_h)

    # check shapes and compute RMS errors
    errors = []
    for stride, X_h in zip(strides, X_sim):
        x_ref_sub = X_ref[::stride]
        if x_ref_sub.shape != X_h.shape:
            raise RuntimeError(
                f"Shape mismatch: ref {x_ref_sub.shape} vs sim {X_h.shape}"
            )
        errors.append(np.sqrt(np.mean((x_ref_sub - X_h) ** 2)))

    print(errors)

    # plot on log-log for convergence rates
    h_plot = h_list.copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(h_plot, errors, "xr")
    ax.loglog(h_list, np.array(h_list) * (errors[0] / h_list[0]), label=f"h^1")
    ax.legend()
    ax.set_xlabel("h")
    ax.set_ylabel("RMS Error")
    plt.show()
