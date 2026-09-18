import numpy as np

# Quadratic bowl  f(x, y) = 0.5 * (a*x**2 + b*y**2).
# Sliders: lr (learning rate = loop gain) and kappa (condition number).
a = 1.0
b = a / kappa
curv = np.array([a, b], dtype="float32")

x = np.array([-3.0, 3.6], dtype="float32")
pts = [x.copy()]
for _ in range(40):
    grad = curv * x          # gradient of the bowl
    x = x - lr * grad        # one gradient-descent step
    pts.append(x.copy())
    if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6:
        break                # diverged

traj = np.ascontiguousarray(np.array(pts), dtype="float32")  # (steps+1, 2)
pole = float(1.0 - lr * a)                                   # stiff-axis closed-loop pole
final_loss = float(0.5 * np.sum(curv * traj[-1] ** 2))
