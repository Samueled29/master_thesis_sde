import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sde.brownian_motion import brownian_motion
from sde.numerical_schemes import euler_maruyama_1d

def get_figures_dir():
    root = Path(__file__).resolve().parents[2]
    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir

def b_holder(x, t):
    return -3 * abs(x) ** (2 / 3)

def b_bdd(x,t):
    return np.sign(x)

def b_regular(x,t):
    return -0.5*x**2

def strong_error_from_diff(diff: np.ndarray, norm_type: str) -> float:
    # diff shape: (M, N_h), M paths and N_h time points
    if norm_type == "space_time_l2":
        # sqrt(E_t,m[|e|^2])
        return float(np.sqrt(np.mean(diff**2)))

    if norm_type == "terminal_l2":
        # sqrt(E_m[|e(T)|^2])
        return float(np.sqrt(np.mean(diff[:, -1] ** 2)))

    if norm_type == "path_sup_l2":
        # (E_m[(sup_t |e|)^2])^(1/2)
        sup_per_path = np.max(np.abs(diff), axis=1)
        return float(np.sqrt(np.mean(sup_per_path**2)))

    raise ValueError(
        "Unknown norm_type. Choose one of: space_time_l2, terminal_l2, path_sup_l2"
    )


def run_convergence(
    M: int = 100, h_ref: float = 1e-5, seed: int = 42, drift_type: str = "holder",norm_type: str = "terminal_l2"
):
    """ # "holder" for holder drift, "bounded" for only bounded (discont) drift,
      "osc" for osciallatory drift, "regual" for regualr drift"""
    t0 = 0.0
    t1 = 1.0
    n_ref = int((t1 - t0) / h_ref) + 1
    times = np.linspace(t0, t1, n_ref)

    h_list = [10 ** (-i) for i in range(1, 5)]
    strides = [int(round(h / h_ref)) for h in h_list]
    strides = [max(1, s) for s in strides]

    rng = np.random.default_rng(seed)

    # M sample BM paths
    W = np.array([brownian_motion(times=times, rng=rng) for _ in range(M)])
   
    eps = 1
    sigma = lambda x, t: np.sqrt(eps)

    drif_dict = {"holder": b_holder,
                 "bounded": b_bdd,
                 "regular": b_regular}
    if drift_type not in drif_dict:
        raise ValueError("Drift not found")
    b = drif_dict[drift_type]

    X_ref = np.array(
        [
            euler_maruyama_1d(b=b, sigma=sigma, times=times, x0=0.0, W=W[i])
            for i in range(M)
        ]
    )
    print(X_ref.shape)

    # compute coarse approximations and errors
    errors = []
    for stride in strides:
        times_h = times[::stride]
        W_h = W[:, ::stride]

        # simulate for each path on the coarse grid
        X_h = np.array(
            [
                euler_maruyama_1d(b=b, sigma=sigma, times=times_h, x0=0.0, W=W_h[i])
                for i in range(M)
            ]
        )

        # compute RMS error averaged over time and ensemble
        diff = X_ref[:, ::stride] - X_h
        err = np.sqrt(np.mean(diff**2))
        errors.append(err)

    h_arr = np.array(h_list, dtype=float)
    err_arr = np.array(errors, dtype=float)

    # fit log-log to get convergence order p
    order = np.argsort(h_arr)
    h_s = h_arr[order]
    err_s = err_arr[order]
    if (h_s <= 0).any() or (err_s <= 0).any():
        raise RuntimeError("h and errors must be > 0 for log-log fit")

    p, intercept = np.polyfit(np.log10(h_s), np.log10(err_s), 1)
    C = 10**intercept
    print(f"Estimated order p = {p:.4f}, C = {C:.4g}")

    # plot data + expected order-1 reference
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(h_s, err_s, "xr", label=f"data ({norm_type})")

    h_line = np.logspace(np.log10(h_s[0]), np.log10(h_s[-1]), 200)

    # shift order-1 line vertically to overlap first data point
    C1 = err_s[0] / h_s[0]
    ax.loglog(h_line, C1 * h_line, "b--", label="order 1 reference: C*h")

    ax.set_xlabel("h")
    ax.set_ylabel("Strong error")
    ax.legend()
    plt.tight_layout()
   
    fig_dir = get_figures_dir()
    fig.savefig(fig_dir / f"plot_b_{drift_type}_eps_{eps}.png", dpi = 300, bbox_inches = "tight")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    run_convergence(M=100, h_ref=1e-5, drift_type = "regular" , norm_type="space_time_l2")
