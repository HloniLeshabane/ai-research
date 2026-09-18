"""
Specialisation — YOUR metric.  (scratch / breadboard — NOT a course module.)

Reproducing the core claim of Jarvis, Lee, Dominé, Saxe & Sarao Mannelli,
"A Theory of Initialisation's Impact on Specialisation" (arXiv:2503.02526,
JSTAT 2025): whether hidden units SPECIALISE (each one serves a single task)
or are REUSED across tasks is controlled by the initialisation — not only by
the data. They report that weight imbalance and high weight entropy push the
network toward specialised solutions.

The setup, the two-head network, the sequential training and the sweep are
given. The MEASUREMENT is yours. That is deliberate: in backprop the
load-bearing mechanism was the gradient. Here it is the instrument. Deciding
what "this unit is specialised" MEANS as a number, and defending that choice,
is the actual skill of interpretability research — a metric you didn't derive
is a result you can't defend in a viva.

THE LOOP
  1. PREDICT — fill in the block below before running anything.
  2. IMPLEMENT specialisation() — replace the zeros with your real metric.
  3. RUN:  python scratch/specialisation.py
     metric_check() scores your metric on hand-built extreme cases where the
     right answer is known by construction. Pass it and the sweep runs.
  4. RECONCILE — then compare your heatmap to the paper. Where it disagrees is
     either your metric, your setup, or their claim. Find out which.

------------------------------------------------------------------------------
YOUR PREDICTION (write before running):
  1. Forgetting vs task similarity: is task A damaged most when task B is
     IDENTICAL to it, ORTHOGONAL to it, or somewhere in between — and why?
  2. Init scale: does a LARGER initial weight scale produce MORE or LESS
     specialisation? (Think about how far a unit has to travel from where it
     started in order to serve one task rather than both.)
  3. If every hidden unit ends up equally engaged by both tasks, what should
     that do to forgetting when you train B after A?
  >
  >
  >
------------------------------------------------------------------------------
"""

import numpy as np


# ============================================================
# GIVEN — two tasks with a controllable similarity knob.
# ============================================================
def two_task_data(n=400, d_in=20, similarity=0.0, seed=0):
    """Teacher-student setup. Two linear teachers separated by an angle.

    similarity = cos(angle) between the teachers:
        1.0  -> identical tasks
        0.0  -> orthogonal tasks
       -1.0  -> opposite tasks
    Returns X, y_a, y_b  with y in {0., 1.}.
    """
    rng = np.random.default_rng(seed)
    theta = np.arccos(np.clip(similarity, -1.0, 1.0))

    w_a = np.zeros(d_in)
    w_a[0] = 1.0
    w_b = np.zeros(d_in)
    w_b[0] = np.cos(theta)
    w_b[1] = np.sin(theta)

    X = rng.standard_normal((n, d_in))
    y_a = (X @ w_a > 0).astype(np.float64)
    y_b = (X @ w_b > 0).astype(np.float64)
    return X, y_a, y_b


# ============================================================
# GIVEN — shared hidden layer, one output head per task.
# Same shape conventions as numpy-nn/nn.py: rows = samples, Z = X @ W + b.
# ============================================================
def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def bce(scores, y):
    N = scores.shape[0]
    p = np.clip(sigmoid(scores), 1e-12, 1 - 1e-12)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return loss, (sigmoid(scores) - y) / N


class TwoHeadNet:
    """d_in -> hidden (relu/tanh) -> two separate scalar heads, 'a' and 'b'.

    The hidden layer is SHARED. That shared layer is where specialisation
    either happens or does not — it is the only thing the two tasks compete
    over.

    init knobs (the independent variables of the whole study):
      scale      — global multiplier on the initial weight magnitudes
      imbalance  — how much larger layer 1 is than the heads at init.
                   >1 puts the mass in the input weights, <1 in the heads.
    """

    def __init__(self, d_in=20, hidden=16, scale=1.0, imbalance=1.0,
                 activation="relu", seed=0):
        rng = np.random.default_rng(seed)
        self.activation = activation
        s1 = scale * np.sqrt(imbalance) / np.sqrt(d_in)
        s2 = scale / (np.sqrt(imbalance) * np.sqrt(hidden))
        self.W1 = rng.standard_normal((d_in, hidden)) * s1
        self.b1 = np.zeros(hidden)
        self.Wh = {t: rng.standard_normal((hidden, 1)) * s2 for t in ("a", "b")}
        self.bh = {t: np.zeros(1) for t in ("a", "b")}

    def _act(self, Z):
        return np.tanh(Z) if self.activation == "tanh" else np.maximum(0.0, Z)

    def _act_bwd(self, dA, Z, A):
        if self.activation == "tanh":
            return dA * (1.0 - A ** 2)
        return dA * (Z > 0).astype(dA.dtype)

    def hidden(self, X):
        """GIVEN. The (N, H) hidden activations — the thing you measure."""
        return self._act(X @ self.W1 + self.b1)

    def forward(self, X, task):
        Z1 = X @ self.W1 + self.b1
        A1 = self._act(Z1)
        scores = (A1 @ self.Wh[task] + self.bh[task]).squeeze(-1)
        return scores, (X, Z1, A1)

    def step(self, X, y, task, lr):
        scores, (X, Z1, A1) = self.forward(X, task)
        loss, dscores = bce(scores, y)
        dS = dscores[:, None]

        dWh = A1.T @ dS
        dbh = dS.sum(axis=0)
        dA1 = dS @ self.Wh[task].T
        dZ1 = self._act_bwd(dA1, Z1, A1)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)

        self.Wh[task] -= lr * dWh
        self.bh[task] -= lr * dbh
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        return loss

    def accuracy(self, X, y, task):
        scores, _ = self.forward(X, task)
        return float(((scores > 0).astype(np.float64) == y).mean())


# ============================================================
# ============================================================
# YOUR JOB.  Everything above and below this block is plumbing.
# ============================================================
#
# CONTRACT
#   inputs
#     Aa : (N, H) hidden activations on inputs drawn from task A's positive set
#     Ab : (M, H) hidden activations on inputs drawn from task B's positive set
#          column j is the same unit in both
#   returns
#     per_unit : (H,) float, each in [0, 1]
#                1.0 = this unit serves exactly one of the two tasks
#                0.0 = this unit is engaged equally by both
#     network  : float, a single summary of the whole hidden layer
#
# TWO HINTS, NO MORE
#   - A unit that responds the same way to both tasks carries no information
#     about WHICH task it is in. A unit that responds to one and is silent for
#     the other carries the maximum.
#   - Your metric must survive metric_check below, which includes a rescaling
#     test and a swap test. Work out what those rule out before you write
#     anything — between them they eliminate the first metric most people
#     reach for.
#
# Derive it. Do not look it up in the paper first: write yours, then compare.
# Where yours differs from theirs is the most interesting paragraph you will
# have to say to a supervisor.
# ============================================================
def specialisation(Aa, Ab):
    H = Aa.shape[1]

    # TODO: replace with your metric.
    per_unit = np.zeros(H)

    network = float(per_unit.mean())
    return per_unit, network


# ============================================================
# GIVEN — your reconcile instrument. The analogue of grad_check.
# Hand-built cases where the answer is known by construction, so the metric is
# graded against ground truth rather than against whether the heatmap looks
# convincing. This is the whole reason you can trust a result later.
# ============================================================
def metric_check(verbose=True):
    rng = np.random.default_rng(1)
    N, H = 200, 8
    results = {}

    # 1. PERFECTLY SPECIALISED: first half of the units fire only on task A,
    #    the rest only on task B. Ground truth: every unit scores 1.
    Aa = np.zeros((N, H))
    Ab = np.zeros((N, H))
    Aa[:, :H // 2] = np.abs(rng.standard_normal((N, H // 2)))
    Ab[:, H // 2:] = np.abs(rng.standard_normal((N, H // 2)))
    pu, net = specialisation(Aa, Ab)
    ok_shape = pu.shape == (H,) and pu.min() >= -1e-9 and pu.max() <= 1 + 1e-9
    results["per_unit shape/range"] = (float(pu.max()), ok_shape)
    results["specialised -> 1.0"] = (net, abs(net - 1.0) < 0.05
                                     and pu.min() > 0.9)

    # 2. PERFECTLY SHARED: identical response to both tasks. Ground truth: 0.
    shared = np.abs(rng.standard_normal((N, H)))
    _, net = specialisation(shared, shared.copy())
    results["shared -> 0.0"] = (net, abs(net) < 0.05)

    # 3. MONOTONIC: interpolate shared -> specialised, the score must rise.
    base = np.abs(rng.standard_normal((N, H)))
    scores = []
    for alpha in np.linspace(0.0, 1.0, 6):
        Aa_i = base.copy()
        Ab_i = base.copy()
        Ab_i[:, :H // 2] *= (1.0 - alpha)   # half the units go quiet on B
        scores.append(specialisation(Aa_i, Ab_i)[1])
    mono = all(scores[i + 1] >= scores[i] - 1e-9 for i in range(len(scores) - 1))
    # must actually RISE, not merely fail to fall — a constant passes the first
    # condition and tells you nothing.
    results["monotonic in mixing"] = (scores[-1] - scores[0],
                                      mono and (scores[-1] - scores[0]) > 0.2)

    # 4. SCALE INVARIANCE: multiplying ALL activations by 10 changes the units,
    #    not the structure. A raw-difference metric dies here.
    Aa2 = np.abs(rng.standard_normal((N, H)))
    Ab2 = np.abs(rng.standard_normal((N, H)))
    s1 = specialisation(Aa2, Ab2)[1]
    s2 = specialisation(Aa2 * 10.0, Ab2 * 10.0)[1]
    results["scale invariant"] = (abs(s1 - s2), abs(s1 - s2) < 1e-6)

    # 5. SYMMETRY: which task you happen to call A is arbitrary.
    s3 = specialisation(Ab2, Aa2)[1]
    results["symmetric in tasks"] = (abs(s1 - s3), abs(s1 - s3) < 1e-6)

    if verbose:
        print("metric_check:")
        for name, (val, ok) in results.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name:24s}  ({val:+.4f})")
    passed = all(ok for _, ok in results.values())
    print("  >>> " + ("metric is sound - running the sweep" if passed
                      else "metric not sound yet - the sweep is gated behind this"))
    return passed


# ============================================================
# GIVEN — the experiment. Sequential (continual) training: task A, then task B.
# ============================================================
def run_one(similarity, scale, imbalance=1.0, hidden=16, d_in=20, n=400,
            steps=400, lr=0.5, seed=0):
    X, y_a, y_b = two_task_data(n=n, d_in=d_in, similarity=similarity, seed=seed)
    net = TwoHeadNet(d_in=d_in, hidden=hidden, scale=scale,
                     imbalance=imbalance, seed=seed)

    acc_a_before = 0.0
    for phase, y in (("a", y_a), ("b", y_b)):
        for _ in range(steps):
            net.step(X, y, phase, lr)
        if phase == "a":
            acc_a_before = net.accuracy(X, y_a, "a")

    # Probe the shared layer with each task's own inputs. Whatever asymmetry
    # the two tasks carved out of that layer is what you are measuring.
    #
    # KNOWN CONFOUND — flagged, not fixed, because the fix is a research
    # decision and it is yours. These two probe sets are different subsets of
    # X, so at similarity = 1.0 they are the SAME subset and the metric is
    # pinned to 0 by construction rather than by anything the network did.
    # Any story you tell about the similarity = 1 column is an artifact of the
    # probe, not a result. Alternatives: probe both heads on identical inputs;
    # or measure each unit's causal contribution by ablating it and reading the
    # damage to each head. The second is closer to what mech interp actually
    # does, and is a better answer if a supervisor asks why you chose this.
    Aa = net.hidden(X[y_a == 1])
    Ab = net.hidden(X[y_b == 1])
    per_unit, spec = specialisation(Aa, Ab)

    return {
        "specialisation": spec,
        "per_unit": per_unit,
        "forgetting": acc_a_before - net.accuracy(X, y_a, "a"),
        "acc_a": net.accuracy(X, y_a, "a"),
        "acc_b": net.accuracy(X, y_b, "b"),
    }


def sweep(similarities=None, scales=None, imbalance=1.0, seeds=(0, 1, 2)):
    """The money figure: specialisation and forgetting over (similarity x scale)."""
    similarities = np.linspace(-1.0, 1.0, 7) if similarities is None else similarities
    scales = np.logspace(-1.5, 0.5, 7) if scales is None else scales

    spec = np.zeros((len(scales), len(similarities)))
    forg = np.zeros((len(scales), len(similarities)))
    for i, sc in enumerate(scales):
        for j, sim in enumerate(similarities):
            runs = [run_one(sim, sc, imbalance=imbalance, seed=s) for s in seeds]
            spec[i, j] = np.mean([r["specialisation"] for r in runs])
            forg[i, j] = np.mean([r["forgetting"] for r in runs])
        print(f"  scale {sc:6.3f}  done")
    return scales, similarities, spec, forg


def plot(scales, similarities, spec, forg, path="scratch/out/specialisation.png"):
    import os
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available - text dump instead)")
        for name, M in (("specialisation", spec), ("forgetting", forg)):
            print(f"\n{name}  (rows = init scale, cols = task similarity)")
            for i, row in enumerate(M):
                print(f"  {scales[i]:6.3f} | " + " ".join(f"{v:5.2f}" for v in row))
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, M, title in zip(axes, (spec, forg), ("specialisation", "forgetting on task A")):
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="magma")
        ax.set_xticks(range(len(similarities)))
        ax.set_xticklabels([f"{s:.2f}" for s in similarities], fontsize=8)
        ax.set_yticks(range(len(scales)))
        ax.set_yticklabels([f"{s:.3f}" for s in scales], fontsize=8)
        ax.set_xlabel("task similarity  (cos angle)")
        ax.set_ylabel("init scale")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    if not metric_check():
        print("\nImplement specialisation() until metric_check passes; "
              "the sweep then runs automatically.")
        raise SystemExit(0)

    print("\nsweeping (similarity x init scale), 3 seeds each:")
    scales, sims, spec, forg = sweep()
    plot(scales, sims, spec, forg)

    print("\nWhat to look for:")
    print("  - a BOUNDARY in the specialisation panel, not a smooth gradient: the")
    print("    paper's claim is a transition, so find where it sits and whether it")
    print("    moves with `imbalance` (run sweep(imbalance=4.0) and diff the two).")
    print("  - whether peak forgetting sits at INTERMEDIATE similarity, and whether")
    print("    it lines up with the specialisation boundary or cuts across it.")
    print("  - your prediction 2, graded.")
