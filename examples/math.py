import os
import numpy as np, matplotlib; matplotlib.use("module://umbruch_matplotlib")
import matplotlib.pyplot as plt
from umbruch_matplotlib import rich

fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
x = np.linspace(0, 2 * np.pi, 300)

ax = axs[0]
ax.plot(x, np.sin(x), label=r"$\sin t$")
ax.plot(x, np.sin(x) + 0.5 * np.cos(2 * x), label=r"$\sin t \pm \frac{1}{2}\cos 2t$")
ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax.set_title("Function names and operators"); ax.legend(fontsize=8)

ax = axs[1]
t = np.linspace(-3, 3, 300)
ax.plot(t, np.exp(-t**2))
ax.set_title(r"Gaussian $e^{-x^2}$, $\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}$", fontsize=10, pad=10)
ax.set_xlabel(r"$x / \sigma$"); ax.set_ylabel(r"$\hat{\rho}_{ij}=\frac{\langle x_i x_j\rangle}{\sigma_i\sigma_j}$")

ax = axs[2]
ax.plot(t, 1 / (1 + t**2), label=rich("Lorentzian <b>L</b>(<i>x</i>) = 1/(1+<i>x</i><sup>2</sup>)"))
ax.legend(fontsize=8, loc="upper right")
ax.set_title(rich("Rich text and <i>math</i> together"))
ax.set_xlabel(rich("rate <i>k</i><sub>on</sub> $(\\mathrm{s}^{-1})$"))

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "math.pdf"))
