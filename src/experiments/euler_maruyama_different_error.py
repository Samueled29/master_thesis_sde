from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama, euler_maruyama_1d


def get_figures_dir():
    root = Path(__file__).resolve().parents[2]
    fig_dir = root / "results" / "figures" / "EM_convergence" 
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def drift_holder(x):
    return 3 * np.abs(x)**(2/3)
    #return np.sqrt(np.abs(x))


def drift_heaviside(x):
    return -np.where(x >= 0, 1.0, -1.0)

def drift_indicator(x):
    # use elementwise logical and for numpy arrays
    return np.where((x >= -2) & (x <= 2), 1.0, 0.0)


def drift_multistep(x):
    points = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    gamma = np.array([0.4, -0.8, 0.8, -0.8, 0.4])

    return sum(g * np.sign(x - xi)
               for g, xi in zip(gamma, points))


def drift_oscillating(x):
    out = np.zeros_like(x, dtype=float)
    #mask = x != 0
    mask = (x != 0.0) & (np.abs(x) <= 1.0)
    out[mask] = np.sin(1.0 / np.abs(x[mask]))
    return out


def drift_irregular(x):
    out = np.zeros_like(x, dtype=float)
    mask = x != 0
    out[mask] = 1/np.abs(x[mask])
    return out

def drift_regular(x):
    return -0.5*x


def euler_from_increments(dW, dt, x0=0.0, sigma=1.0):
    M, N = dW.shape
    X = np.zeros((M,N+1))
    X[:,0] = x0

    for n in range(N):
        X[:,n+1] = X[:,n] + drift_holder(X[:,n]) * dt + sigma * dW[:, n]

    return X


if __name__ == "__main__":
    drift_type = "holder_drift"

    t0 = 0.0
    t1 = 1
    x0 = 1.0

    ks = np.arange(4, 14)
    dt_values = 2.0 ** (-ks)

    k_ref = 17
    dt_ref = 2.0 ** (-k_ref)
    N_ref = 2**k_ref
    M = 2000

    rng = np.random.default_rng(42)
    dW_ref = np.sqrt(dt_ref) * rng.standard_normal((M, N_ref))

    eps = 1.0

    X_ref = euler_from_increments(dW_ref, dt_ref, x0, eps)

    errors = []

    for dt in dt_values:
        N = int(1 / dt)
        m = N_ref // N

        dW_coarse = dW_ref.reshape(M, N, m).sum(axis=2)

        X_ref_coarse = X_ref[:, ::m]

        X_coarse = euler_from_increments(dW_coarse, dt, x0, eps)

        diff = X_coarse - X_ref_coarse  
        l2_at_each_tk = np.sqrt(np.mean(diff**2, axis=0))
        err = np.max(l2_at_each_tk)
        errors.append(err)

    slope, intercept = np.polyfit(np.log(dt_values), np.log(errors), 1)

    print("dt:", dt_values)
    print("errors:", errors)
    print("estimated order:", slope)

    errors = np.array(errors)

plt.figure()
plt.loglog(dt_values, errors, "o-", label="Errors")
plt.loglog(dt_values, np.exp(intercept) * dt_values**slope, "--", label=f"Fitted slope ≈ {slope:.3f}")
plt.loglog(dt_values, dt_values/dt_values[-1]*errors[-1], "-.", label="Order 1 slope reference") 
plt.xlabel(r"$\Delta t$")
plt.ylabel("Error")
plt.title("Strong error Euler-Maruyama")
plt.grid(True, which="both")
plt.legend()

fig_dir = get_figures_dir()

plt.savefig(fig_dir / f"{drift_type}.png", dpi = 300)
plt.show()
plt.close()

