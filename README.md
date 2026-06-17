# ai-research — a lab for learning AI by seeing it

This repo is a **lab**, not a course. The goal isn't to finish lessons; it's to build
mechanistic intuition for the ~10 ideas that are actually load-bearing in modern ML, so they
can be wielded correctly inside real systems (Connex, FlowForge) instead of cargo-culted.

Learning happens *with* Claude Code as the apparatus: it removes friction (setup, plotting,
API lookup) and runs experiments; you keep the cognition (predictions, derivations, the core
implementations). The git history is the record of the journey. `LAB_NOTEBOOK.md` is the
continuity layer between short sessions — predictions in, reconciliations out.

## The one rule

Defeat the **illusion of understanding**: an explanation that *feels* clear but can't be
re-derived. Comprehension is not competence. Every module is built to force the gap between
what you predict and what actually happens out into the open.

## The five loops (how every module is run)

1. **Predict → run → reconcile.** Write the prediction in the notebook *before* running anything.
2. **Implement, then critique.** You write the core mechanism from scratch; Claude diffs it against the canonical version and attacks yours.
3. **Teach it back.** Explain the idea to Claude as if teaching; it probes for the holes.
4. **Socratic mode.** Tell Claude to withhold answers and ask leading questions instead.
5. **Sabotage and predict the failure.** Break init / normalization / the learning rate; predict the symptom; confirm it on the plot.

## What's yours vs. what's Claude's

- **Yours (productive struggle, never outsourced):** predictions, derivations, and from-scratch implementations of the load-bearing ideas — backprop, attention, a tiny GPT.
- **Claude's (unproductive friction, always outsourced):** environment, plotting/instrument code, boilerplate, API lookups, refactors, and read-and-probe walkthroughs of everything that isn't load-bearing.

This is "implement the seams, probe the rest" applied to learning.

## The arc (fundamentals → frontier)

0. **The lab itself** — instruments, notebook, the loops. ← you are here
1. **Optimization as a control loop** — gradient descent, learning-rate stability. *(module 01)*
2. **MLPs + backprop from scratch** — gradient flow, vanishing/exploding, init, normalization.
3. **CNNs** — convolution you already know; learned filters, feature maps, spectral bias.
4. **Embeddings → attention → a tiny GPT** — built from scratch, trained, watched learning.
5. **The modern LLM stack** — pretraining / fine-tuning, scaling laws, inference, quantization.

### North Star: mechanistic interpretability

The destination, flagged on day one. Reverse-engineering the circuits inside a trained network —
probing internal nodes to recover the computation it's actually doing. It's EE reverse-engineering
applied to neural nets, and it's the thing that turns "I understand ML" into "I can debug a
model's reasoning." Every earlier module is chosen to build toward it.

## EE → ML bridges (lead with what you already own)

| ML concept | What it already is, in EE terms |
|---|---|
| Gradient descent + learning rate | Discrete feedback loop; η is loop gain; divergence is instability |
| Momentum / Adam | Low-pass filtering the gradient; adaptive gain |
| Convolutional layers | Convolution / impulse response / FIR filters |
| Attention | Data-dependent / matched filtering |
| Cross-entropy, KL divergence | Shannon information theory |
| Regularization, dropout | Noise injection, SNR, damping |
| Quantization / mixed precision | Fixed-point arithmetic; precision vs. dynamic range |
| Spectral bias of networks | Frequency-domain analysis; nets learn low frequencies first |

## Running things

Python 3.12 is installed. From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python modules\01_gradient_descent_as_control_loop\gradient_descent.py
```

Module 01 also ships an interactive playground that renders in the Claude Code chat — turn the
learning-rate knob and watch the loop go stable → underdamped → unstable.
