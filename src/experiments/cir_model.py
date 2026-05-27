import matplotlib.pyplot as plt
import numpy as np

from sde.numerical_schemes import euler_maruyama, euler_maruyama_1d

if __name__ == "__main__":
    M = 10
    b = lambda x, t: -1 * x  # noqa: E731
    g = lambda x, t: 0.5  # noqa: E731

    times = np.linspace(0, 10, 1000)
    x0 = 0

    # W = brownian_motion(times)
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.set_xlabel("time")
    X_multi = np.zeros((2 * M, len(times)))
    for i in range(M):
        X = euler_maruyama_1d(b, g, times, x0)
        X_multi[i, :] = X.flatten()

        axs.plot(times, X_multi[i, :])
    for i in range(M):
        x0_arr = np.array([x0])
        X = euler_maruyama(b=b,sigma =  g, times = times, x0 =x0_arr, n= 1, k = 1)
        X_multi[M + i, :] = X.flatten()

        axs.plot(times, X_multi[M + i, :])
    plt.show()
    # print(X_multi.shape)
