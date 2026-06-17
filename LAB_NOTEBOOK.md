# Lab notebook

The continuity layer between short sessions — this is how they connect. The rule:
**predictions go in before you run anything.** Reconcile after. The gap is the learning.

Template per module:
- **Predict** — what will happen, and *why* (mechanism, not vibe).
- **Run** — what actually happened.
- **Reconcile** — where prediction ≠ reality, and the corrected mental model.
- **Teach-back** — explain it in 3 sentences, from memory.

---

## Module 01 — gradient descent as a control loop
Date: 2026-06-17 · Status: open

### Before you touch the slider or the code — predict
Setup: minimizing `f(x, y) = ½(a·x² + b·y²)`, a quadratic bowl. Plain gradient descent, step size η.

1. For the 1-D case `f(x) = ½·a·x²`, write the update as `x_{k+1} = (____) · x_k`. What is the multiplier?
2. For what range of η does `x_k → 0` (converge)? Write the inequality. This is the stability boundary.
3. At η just below that boundary, what does the path *look like* — smooth, or something else? Why?
4. Now 2-D with `a ≠ b` (condition number κ = a/b). Which curvature sets the stability limit — the stiff axis or the soft one? Which one sets how *slow* convergence is?
5. Predict the exact η at which it diverges (κ = 8, a = 1), and sketch the path shape just below it.

> Your answers (write them here first):
>
> 1.
> 2.
> 3.
> 4.
> 5.

### Run
Play the chat playground and run `gradient_descent.py` (it prints a sweep + draws four regimes).
> What it actually showed:

### Reconcile
> Where you were wrong, and the fixed mental model:

### Teach-back (3 sentences, from memory)
>

### The bridge — read only after attempting 1–5
η is loop gain. The per-step multiplier you found in (1) is the closed-loop pole; `|pole| < 1`,
inside the unit circle, is exactly the stability condition from (2). Underdamped zig-zag is a
real, negative pole; divergence is the pole leaving the unit circle; the slow crawl at high κ is
a different pole sitting too close to 1. The whole module is that one picture, made visible.
