import numpy as np

# Inputs — edit these freely (the editor re-runs when you pause typing).
# Weights w0..w2 and the bias are the sliders on the right.
X = np.array([0.8, 0.5, 0.2], dtype="float32")
W = np.array([w0, w1, w2], dtype="float32")

# Per-input signed contribution, and the neuron's linear output.
contributions = (X * W).astype("float32")
activation = float(X @ W + bias)
