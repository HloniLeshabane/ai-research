"""Module 01 - Gradient descent as a control loop.

Minimize a quadratic bowl  f(x) = 1/2 * x^T diag(a, b) x  with plain gradient
descent, and watch the learning rate eta act as loop gain:

    x_{k+1} = x_k - eta * grad f(x_k) = (I - eta * diag(a, b)) x_k

Per axis the iteration is  x_{k+1} = (1 - eta * curvature) * x_k. The factor in
parentheses is the closed-loop pole. |pole| < 1 (inside the unit circle) is the
stability condition - exactly like a discrete-time feedback system.

EE framing: eta is gain. Too little -> sluggish (overdamped). More -> fast. Past
a point -> overshoot and oscillation (underdamped). Past the unit circle -> the
loop is unstable and the loss explodes.

Run:  python gradient_descent.py
Then do the EXERCISES at the bottom - that is where the learning is.
"""

import numpy as np
import matplotlib.pyplot as plt


def gd_step(x, curv, eta):
    """One gradient-descent step on f(x) = 1/2 * sum(curv * x**2).

    EXERCISE 1 asks you to delete this body and re-derive it from grad f. The
    gradient of 1/2 * curv_i * x_i**2 is curv_i * x_i.
    """
    grad = curv * x
    return x - eta * grad


def descend(curv, eta, steps=60, x0=None):
    """Run gradient descent; return the trajectory, shape (<=steps+1, dim)."""
    x = np.array(x0 if x0 is not None else [-3.0, 3.6], dtype=float)
    traj = [x.copy()]
    for _ in range(steps):
        x = gd_step(x, curv, eta)
        traj.append(x.copy())
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6:
            break  # diverged
    return np.array(traj)


def loss(curv, x):
    return 0.5 * np.sum(curv * np.asarray(x) ** 2)


def regime(curv, eta):
    """Classify a run from the analytic pole on the stiff axis.

    You predict the 2/a boundary in the notebook first; this just confirms it.
    """
    pole = 1.0 - eta * np.max(curv)
    mag = abs(pole)
    if mag > 1.0:
        return "unstable", pole
    if mag < 1e-9:
        return "well-tuned", pole
    if pole < 0:
        return "underdamped", pole
    return "overdamped", pole


def sweep(kappa=8.0):
    """Empirically find the divergence boundary - reconcile with your predicted 2/a."""
    a = 1.0
    curv = np.array([a, a / kappa])
    print(f"\ncondition number kappa = {kappa:.0f}, stiff curvature a = {a}")
    print(f"{'eta':>6} {'final loss (60 steps)':>24} {'regime':>14}")
    for eta in np.arange(0.2, 2.41, 0.2):
        traj = descend(curv, float(eta))
        fl = loss(curv, traj[-1])
        name, _ = regime(curv, float(eta))
        shown = "diverged" if (not np.isfinite(fl) or fl > 1e6) else f"{fl:.2e}"
        print(f"{eta:>6.2f} {shown:>24} {name:>14}")


def plot_regimes(kappa=8.0):
    """Four learning rates side by side: watch the path change shape as eta grows."""
    a = 1.0           # stiff-axis curvature -> the stability boundary lives here
    b = a / kappa     # soft-axis curvature  -> sets how slow convergence is
    curv = np.array([a, b])
    etas = [0.4, 1.0, 1.7, 2.1]   # one per regime; you predicted where the boundary is

    xs = np.linspace(-4.5, 4.5, 240)
    X, Y = np.meshgrid(xs, xs)
    Z = 0.5 * (a * X ** 2 + b * Y ** 2)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, eta in zip(axes, etas):
        ax.contour(X, Y, Z, levels=12, colors="0.7", linewidths=0.6)
        traj = descend(curv, eta)
        name, pole = regime(curv, eta)
        diverged = (not np.all(np.isfinite(traj[-1]))) or np.max(np.abs(traj[-1])) > 1e6
        color = "#E24B4A" if diverged else "#1D9E75"
        t = np.clip(traj, -4.5, 4.5)
        ax.plot(t[:, 0], t[:, 1], "-o", color=color, ms=3, lw=1.2)
        ax.plot(0, 0, "x", color="#BA7517", ms=10, mew=2)
        ax.set_title(f"eta = {eta}   |pole| = {abs(pole):.2f}\n{name}", fontsize=11)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Gradient descent on a quadratic bowl (kappa = {kappa:.0f})", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    sweep()
    plot_regimes()


# ---------------------------------------------------------------------------
# EXERCISES (the actual point - do these in the notebook AND here)
#
#  1. Delete the body of gd_step and re-derive it from grad f. Confirm the run
#     is identical.
#  2. You predicted a divergence boundary in the notebook (eta < 2/a). Run
#     sweep() and reconcile: does the empirical boundary match? At kappa = 8,
#     a = 1, what exact eta is it?
#  3. Crank kappa to 40 and rerun plot_regimes(40). The stable runs still
#     converge - but watch the soft axis. Why does the path crawl?
#     (slow-axis pole 1 - eta/kappa sits near 1)
#  4. Add momentum:  v = beta * v + grad ;  x -= eta * v  (heavy ball). Show it
#     lets you push eta past 2 and damps the kappa = 40 zig-zag. What does beta
#     do, in filter terms?
#  5. Sabotage: start far out on one axis and predict, before running, which
#     axis dies first and which lingers.
# ---------------------------------------------------------------------------
