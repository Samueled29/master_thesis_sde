import matplotlib.pyplot as plt
import numpy as np

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama, euler_maruyama_1d

if __name__ == "__main__":
    t0 = 0
    t1 = 100
    n = 1000
    times = np.linspace(t0, t1, n)
    eps = 4 * 100
    x0 = 0

    M = 20

    rng = np.random.default_rng(42)

    # b = lambda x,t: -np.sqrt(abs(x))
    b = lambda x, t: -3 * abs(x) ** (2 / 3)
    sigma = lambda x, t: np.sqrt(eps)

    X_sim = np.zeros((M, len(times)))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    for i in range(M):
        W = brownian_motion(times, rng)
        X_sim[i, :] = euler_maruyama_1d(
            b=b, sigma=sigma, times=times, x0=x0, W=W, rng=None
        )
        ax.plot(times, X_sim[i, :])

    ax.set_ylabel("X_t")
    ax.set_xlabel("Y_t")
    mean_path = X_sim.mean(axis=0)
    std_path = X_sim.std(axis=0)
    ax.plot(times, mean_path, color="k", linewidth=2)
    ax.fill_between(times, mean_path - std_path, mean_path + std_path, alpha=0.6)

    # histogram of endpoints
    ax2.hist(X_sim[:, -1], bins=20)
    ax2.set_xlabel("X_T (endpoints)")
    ax2.set_ylabel("frequency")

    plt.tight_layout()
    plt.show()
