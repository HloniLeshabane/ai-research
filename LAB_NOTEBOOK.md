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

---

## Module — autograd from scratch (micrograd)
Date: 2026-07-08 · Status: open

Companion to Karpathy's "spelled-out intro to neural networks and backpropagation".
The lab module replays one traced epoch of a 2→3→1 tanh net learning XOR, one scalar
event at a time: forward → loss → zero_grad → backward → update.

### Before you press step — predict
1. `d = pred − y`, `L = d·d` (one mul, same node in both slots). Write ∂L/∂pred by hand.
   For pred = −0.48, y = −1, what number should the debugger show on the `pred` node's `g`
   when backprop reaches it?
2. An add node has two inputs. What does backprop through `+` do to the incoming gradient,
   and why is the `+=` (not `=`) in every `_backward` load-bearing? Name the exact graph
   shape where `=` gives the wrong answer.
3. A tanh node has data ≈ ±0.99. Roughly what is its local gain 1 − t²? What does that do
   to every gradient upstream of it?
4. Trace epoch 0, then trace a late epoch (say 60). Predict how the grads on the weight
   nodes differ, and what the loss-curve marker will show about why.
5. Sabotage: in `__mul__`'s `_backward`, change both `+=` to `=`. L = d·d puts d in both
   slots of one mul, so its gradient comes out as d instead of 2d. Training still converges —
   predict what changes, and predict whether the gradcheck badge catches it. Then explain
   why the same edit in `__add__` would change nothing on this graph.

> Your answers (write them here first):
>
> 1.
> 2.
> 3.
> 4.
> 5.

### Run
Step through iteration 1 of epoch 0 in full. Then use ⏭ to spot-check your predictions.
> What it actually showed:

### Reconcile
> Where you were wrong, and the fixed mental model:

### Teach-back (3 sentences, from memory)
>

### The bridge — read only after attempting 1–5
The graph is a signal-flow graph and backprop is the adjoint network: same topology, edges
reversed, each op replaced by its small-signal gain (× → the other input, tanh → 1 − t²,
+ → unity). The `+=` at merge points is superposition. A saturated tanh is a stage biased
into the rails — gain ≈ 0, and the error signal can't propagate past it.
