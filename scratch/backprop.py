"""
Backprop — YOUR implementation.  (scratch / breadboard — NOT a course module.)

The forward pass, the loss, and a numerical gradient-checker are given. The
backward pass is yours to derive and write. That is the load-bearing mechanism;
the whole point is that YOU implement it.

THE LOOP
  1. PREDICT — fill in the block below before running anything.
  2. IMPLEMENT backward() — replace the zeros with the real gradients.
  3. RUN:  python scratch/backprop.py
     The grad-check compares your analytical gradients to finite-difference
     numerical ones, per parameter. < 1e-6 means your backward pass is correct.
  4. RECONCILE — if it's wrong, find WHICH layer is off and why. Don't read the
     checker to derive the math; use it to grade your derivation.

When it passes, hand this back and it gets wrapped into module #5 with a
gradient-flow visualization — dressing up YOUR mechanism, not Claude's.

------------------------------------------------------------------------------
YOUR PREDICTION (write before running):
  1. Gradient flow: at init, do you expect dW1 (first layer) or dW2 (last layer)
     to have the larger magnitude — and why? (This is the gradient-flow question
     the eventual visualization will show.)
  2. If you forget the tanh derivative (1 - a1**2) in dz1, will the grad-check
     fail on all four params, or only some? Which?
  3. After ~400 steps on XOR, what should the loss do — and could it get stuck?
  >
  >
  >
------------------------------------------------------------------------------
"""

import numpy as np

rng = np.random.default_rng(0)


# A tiny MLP:  2 inputs -> 4 hidden (tanh) -> 1 output (sigmoid), BCE loss.
def init():
    return {
        "W1": rng.normal(0, 0.5, (2, 4)), "b1": np.zeros(4),
        "W2": rng.normal(0, 0.5, (4, 1)), "b2": np.zeros(1),
    }


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(p, X):
    """GIVEN. Returns the prediction plus a cache of intermediates for backward()."""
    z1 = X @ p["W1"] + p["b1"]      # (N, 4)
    a1 = np.tanh(z1)                # (N, 4)
    z2 = a1 @ p["W2"] + p["b2"]     # (N, 1)
    a2 = sigmoid(z2)               # (N, 1)  prediction
    return a2, {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}


def loss(a2, y):
    eps = 1e-9
    return float(-np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps)))


# ----------------------------------------------------------------------------
# YOUR JOB: implement the backward pass. Return grads with the SAME keys as p.
# The chain rule is all you need. Two hints, no more:
#   - for sigmoid output + BCE loss, d loss / d z2  simplifies to (a2 - y) / N
#   - tanh'(z1) = 1 - a1**2
# ----------------------------------------------------------------------------
def backward(p, cache, y):
    X, a1, a2 = cache["X"], cache["a1"], cache["a2"]
    N = X.shape[0]

    # TODO: replace each zero with the real gradient. Start at the output (dz2)
    # and work backwards toward the inputs.
    dz2 = np.zeros_like(a2)        # d loss / d z2     -> shape (N, 1)
    dW2 = np.zeros_like(p["W2"])   # d loss / d W2     -> shape (4, 1)
    db2 = np.zeros_like(p["b2"])   # d loss / d b2     -> shape (1,)
    dz1 = np.zeros_like(a1)        # d loss / d z1     -> shape (N, 4)
    dW1 = np.zeros_like(p["W1"])   # d loss / d W1     -> shape (2, 4)
    db1 = np.zeros_like(p["b1"])   # d loss / d b1     -> shape (4,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ----------------------------------------------------------------------------
# GIVEN — your reconcile instrument. Finite-difference gradient check.
def numerical_grad(p, X, y, key, eps=1e-5):
    g = np.zeros_like(p[key])
    it = np.nditer(p[key], flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = p[key][idx]
        p[key][idx] = orig + eps
        lp = loss(forward(p, X)[0], y)
        p[key][idx] = orig - eps
        lm = loss(forward(p, X)[0], y)
        p[key][idx] = orig
        g[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    return g


def grad_check(p, X, y):
    a2, cache = forward(p, X)
    analytic = backward(p, cache, y)
    print(f"  initial loss: {loss(a2, y):.4f}")
    worst = 0.0
    for key in ("W1", "b1", "W2", "b2"):
        num = numerical_grad(p, X, y, key)
        rel = np.max(np.abs(analytic[key] - num)) / (np.max(np.abs(num)) + 1e-12)
        worst = max(worst, rel)
        print(f"  {key}: max rel error = {rel:.2e}")
    verdict = "CORRECT" if worst < 1e-6 else "not yet — find the layer that is off"
    print(f"  >>> worst error {worst:.2e}  ->  {verdict}")
    return worst


def train(p, X, y, steps=400, lr=0.5):
    for i in range(steps):
        a2, cache = forward(p, X)
        grads = backward(p, cache, y)
        for k in p:
            p[k] -= lr * grads[k]
        if i % 50 == 0:
            print(f"  step {i:4d}  loss {loss(a2, y):.4f}")
    print(f"  final preds: {forward(p, X)[0].ravel().round(3)}  (target: {y.ravel()})")


if __name__ == "__main__":
    # XOR — not linearly separable, so it genuinely needs the hidden layer.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    p = init()
    print("grad check (implement backward() first):")
    err = grad_check(p, X, y)

    if err < 1e-6:
        print("\ngradients correct — training on XOR:")
        train(p, X, y)
    else:
        print("\nfix backward() until the grad-check passes, then training runs automatically.")
