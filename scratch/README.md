# scratch — your breadboard

This is where **you** implement the load-bearing mechanisms from scratch, before any
course module is built around them. The rule for this lab:

- **The ~10 ideas that carry weight** (backprop, attention, a tiny GPT, mechanistic
  interpretability): you write the numpy mechanism here first. Predict the result, run,
  reconcile. Only once *your* version works does it get wrapped into a course module —
  Claude does the Pyodide + SVG plumbing; you did the neuron.
- **Read-and-probe everything else** (vectors, activation functions, …): Claude can build
  those directly.

If a file here leaves a mechanism stubbed with a `TODO`, that's the point — don't ask for
the answer, derive it. The prediction prompts and grad-checks are your reconcile step.

Start with `backprop.py`.

## Files

- **`backprop.py`** — the gradient. Forward + grad-check given; `backward()` is yours.
- **`specialisation.py`** — the *instrument*. Reproduces the central claim of
  Jarvis et al., *A Theory of Initialisation's Impact on Specialisation*
  (arXiv:2503.02526): that initialisation, not just data, decides whether hidden
  units specialise or get reused across tasks. Two-task setup, two-head network,
  sequential training and the sweep are given; the `specialisation()` metric is
  yours to derive. `metric_check()` grades it against hand-built cases where the
  answer is known by construction — the analogue of the grad-check. Groundwork for
  the Wits MSc application; see the notebook.
