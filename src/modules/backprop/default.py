# ============================================================================
#  YOUR nn.py — the mechanism, running in the browser (via Pyodide).
#  Vendored snapshot of Documents/numpy-nn/nn.py. This is the real code you
#  wrote; edit it here to experiment (break tanh_backward and watch the
#  boundary fail to form). The canonical copy lives in the numpy-nn project.
# ============================================================================
import numpy as np


# ---- affine layer ----------------------------------------------------------
def affine_forward(X, W, b):
    Z = X @ W + b
    return Z, X


def affine_backward(dZ, cache, W):
    X = cache
    dW = X.T @ dZ          # (D_in, D_out)
    db = dZ.sum(axis=0)    # (D_out,)
    dX = dZ @ W.T          # (N, D_in)
    return dX, dW, db


# ---- activations -----------------------------------------------------------
def tanh_forward(Z):
    A = np.tanh(Z)
    return A, A


def tanh_backward(dA, cache):
    A = cache
    return dA * (1.0 - A ** 2)


def relu_forward(Z):
    A = np.maximum(0, Z)
    return A, Z


def relu_backward(dA, cache):
    Z = cache
    return dA * (Z > 0).astype(dA.dtype)


# ---- loss: sigmoid + binary cross-entropy ----------------------------------
def sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def bce_loss(scores, y):
    N = scores.shape[0]
    p = sigmoid(scores)
    p_clipped = np.clip(p, 1e-12, 1 - 1e-12)
    loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
    dscores = (p - y) / N        # d(BCE)/d(logit) collapses to (p - y)/N
    return loss, dscores


# ---- the network -----------------------------------------------------------
class Network:
    def __init__(self, layer_sizes, activation="tanh", seed=0):
        rng = np.random.default_rng(seed)
        self.activation = activation
        self.params = {}
        self.caches = {}
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            self.params[f"W{i+1}"] = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            self.params[f"b{i+1}"] = np.zeros(fan_out)
        self.n_layers = len(layer_sizes) - 1

    def forward(self, X):
        A = X
        act_fwd = tanh_forward if self.activation == "tanh" else relu_forward
        for i in range(1, self.n_layers):
            W, b = self.params[f"W{i}"], self.params[f"b{i}"]
            Z, cache_aff = affine_forward(A, W, b)
            A, cache_act = act_fwd(Z)
            self.caches[f"aff{i}"] = (cache_aff, W)
            self.caches[f"act{i}"] = cache_act
        W, b = self.params[f"W{self.n_layers}"], self.params[f"b{self.n_layers}"]
        Z, cache_aff = affine_forward(A, W, b)
        self.caches[f"aff{self.n_layers}"] = (cache_aff, W)
        return Z.squeeze(-1)

    def backward(self, dscores):
        grads = {}
        act_bwd = tanh_backward if self.activation == "tanh" else relu_backward
        dZ = dscores[:, None]
        i = self.n_layers
        cache_aff, W = self.caches[f"aff{i}"]
        dA_prev, dW, db = affine_backward(dZ, cache_aff, W)
        grads[f"W{i}"], grads[f"b{i}"] = dW, db
        for i in range(self.n_layers - 1, 0, -1):
            dZ = act_bwd(dA_prev, self.caches[f"act{i}"])
            cache_aff, W = self.caches[f"aff{i}"]
            dA_prev, dW, db = affine_backward(dZ, cache_aff, W)
            grads[f"W{i}"], grads[f"b{i}"] = dW, db
        return grads

    def update(self, grads, lr):
        for k in self.params:
            self.params[k] -= lr * grads[k]

    def predict(self, X):
        return (sigmoid(self.forward(X)) >= 0.5).astype(int)


# ============================================================================
#  PLUMBING — datasets (2-D so the boundary is visible). Not load-bearing.
# ============================================================================
def _shuffle(X, y, rng):
    idx = rng.permutation(len(X))
    return X[idx].astype(np.float64), y[idx].astype(np.float64)


def make_xor(n=120, noise=0.10, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    labels = np.array([0, 1, 1, 0])
    X, y = [], []
    for c, l in zip(centers, labels):
        X.append(c + noise * rng.standard_normal((n, 2)))
        y.append(np.full(n, l))
    return _shuffle(np.vstack(X), np.concatenate(y), rng)


def make_moons(n=300, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n_out = n // 2
    n_in = n - n_out
    t_o = np.linspace(0, np.pi, n_out)
    outer = np.stack([np.cos(t_o), np.sin(t_o)], axis=1)
    t_i = np.linspace(0, np.pi, n_in)
    inner = np.stack([1 - np.cos(t_i), 1 - np.sin(t_i) - 0.5], axis=1)
    X = np.vstack([outer, inner]) + noise * rng.standard_normal((n, 2))
    y = np.concatenate([np.zeros(n_out), np.ones(n_in)])
    return _shuffle(X, y, rng)


def make_spirals(n=150, noise=0.20, turns=1.5, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls in (0, 1):
        r = np.linspace(0.05, 1.0, n)
        theta = np.linspace(0, turns * 2 * np.pi, n) + cls * np.pi
        theta += noise * rng.standard_normal(n)
        X.append(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1))
        y.append(np.full(n, cls))
    return _shuffle(np.vstack(X), np.concatenate(y), rng)


# ============================================================================
#  DRIVER — train your net, snapshotting the decision boundary as it learns.
#  Sliders (dataset / activation / hidden / lr / epochs) arrive as globals.
# ============================================================================
DATASETS = [make_xor, make_moons, make_spirals]
DATASET_NAMES = ["XOR", "moons", "spirals"]
ACTS = ["tanh", "relu"]

d_i = int(dataset)
a_i = int(activation)
H = int(hidden)
E = int(epochs)

X, y = DATASETS[d_i](seed=0)
dataset_name = DATASET_NAMES[d_i]
act_name = ACTS[a_i]

# A grid over the input plane; we run the net on it to shade the boundary.
R = 64
pad = 0.5
xmin, xmax = X[:, 0].min() - pad, X[:, 0].max() + pad
ymin, ymax = X[:, 1].min() - pad, X[:, 1].max() + pad
gx = np.linspace(xmin, xmax, R)
gy = np.linspace(ymin, ymax, R)
GX, GY = np.meshgrid(gx, gy)                     # 'xy': GX[r,c]=gx[c], GY[r,c]=gy[r]
grid = np.stack([GX.ravel(), GY.ravel()], axis=1)

model = Network([2, H, 1], activation=act_name, seed=0)

# ---- LIVE GRADIENT CHECK ----------------------------------------------------
# Does YOUR backward() agree with finite differences (dL/dw ~ [L(w+e)-L(w-e)]/2e)?
# This runs on every edit. Break tanh_backward above (e.g. drop the 1 - A**2) and
# watch this flip to FAIL — and the boundary below stops forming. Analytic and
# numerical should agree to ~1e-7 in float64; a real bug shows up as ~1e-1 or ~1.
_Xc, _yc = X[:8], y[:8]                           # tiny fixed batch


def _gc_loss():
    return bce_loss(model.forward(_Xc), _yc)[0]


_s = model.forward(_Xc)
_, _dsc = bce_loss(_s, _yc)
_analytic = model.backward(_dsc)                  # <- gradients from YOUR backward

_eps = 1e-6
gradcheck_max_err = 0.0
gradcheck_worst = ""
for _name, _P in model.params.items():
    _num = np.zeros_like(_P)
    _it = np.nditer(_P, flags=["multi_index"], op_flags=["readwrite"])
    while not _it.finished:
        _ix = _it.multi_index
        _o = _P[_ix]
        _P[_ix] = _o + _eps; _lp = _gc_loss()
        _P[_ix] = _o - _eps; _lm = _gc_loss()
        _P[_ix] = _o                              # restore (params end up at init)
        _num[_ix] = (_lp - _lm) / (2 * _eps)
        _it.iternext()
    _rel = np.abs(_analytic[_name] - _num) / np.maximum(np.abs(_analytic[_name]) + np.abs(_num), 1e-12)
    _m = float(_rel.max())
    if _m > gradcheck_max_err:
        gradcheck_max_err = _m
        gradcheck_worst = _name
gradcheck_ok = bool(gradcheck_max_err < 1e-4)
gradcheck_max_err = float(gradcheck_max_err)

K = 40                                           # snapshots to animate through
stride = max(1, E // K)
frames, frame_epochs, loss_hist = [], [], []


def snapshot(ep):
    p = sigmoid(model.forward(grid)).reshape(R, R)   # class-1 probability
    frames.append(p.astype(np.float32))
    frame_epochs.append(ep)


snapshot(0)                                      # untrained boundary
for ep in range(1, E + 1):
    scores = model.forward(X)                    # <-- refreshes caches each epoch
    loss, dscores = bce_loss(scores, y)
    grads = model.backward(dscores)
    model.update(grads, lr)
    loss_hist.append(float(loss))
    if ep % stride == 0 or ep == E:
        snapshot(ep)

pred = model.predict(X)
final_acc = float((pred == y).mean())
final_loss = float(loss_hist[-1]) if loss_hist else float("nan")
print(f"{dataset_name} · {act_name} · H={H} · lr={lr} · "
      f"final loss={final_loss:.4f} acc={final_acc:.3f}")

# ---- outputs (all float32 + C-contiguous for the array bridge) -------------
Xdata = np.ascontiguousarray(X, dtype=np.float32)
ydata = np.ascontiguousarray(y, dtype=np.float32)
frames = np.ascontiguousarray(np.stack(frames), dtype=np.float32)   # (K, R, R)
frame_epochs = np.ascontiguousarray(np.array(frame_epochs), dtype=np.float32)
loss_hist = np.ascontiguousarray(np.array(loss_hist), dtype=np.float32)
extent = np.ascontiguousarray(np.array([xmin, xmax, ymin, ymax]), dtype=np.float32)
