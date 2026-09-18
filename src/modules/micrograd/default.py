# ============================================================================
#  micrograd — backprop at scalar granularity, spelled out (after Karpathy's
#  "The spelled-out intro to neural networks and backpropagation").
#
#  The Value class below is YOURS — the load-bearing mechanism. Every op
#  records what built it, and its _backward() is the chain rule for that one
#  op. The debugger on the right replays a traced epoch event by event:
#  forward → loss → backward → update.
#
#  Sabotage ideas (predict the symptom first, then look):
#    · in tanh's _backward, drop the (1 - t*t): gradcheck badge flips red
#    · in __mul__'s _backward, change += to =: L = d·d feeds d into BOTH
#      slots of one mul, so its grad halves (d instead of 2d) — training
#      silently runs at half speed and only the gradcheck can tell you
#    · in __add__'s _backward, delete the other.grad line: the b/c leaves
#      and the h₂/h₃ subtrees go gradient-dead — watch their g pins freeze
#      at 0 and the badge flip red
# ============================================================================
import json
import math


# ---- the engine (load-bearing — this is the whole point) -------------------
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        # tuple, not set (Karpathy uses set(_children)): deterministic order
        # keeps the debugger's node ids stable from run to run
        self._prev = tuple(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other:g}")
        out = Value(self.data + other.data, (self, other), "add")

        def _backward():
            self.grad += 1.0 * out.grad  # d(a+b)/da = 1: adds route grads through
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other:g}")
        out = Value(self.data * other.data, (self, other), "mul")

        def _backward():
            self.grad += other.data * out.grad  # d(ab)/da = b — the other factor
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        # primitive here for a smaller graph; Karpathy composes a + (-1)*b
        other = other if isinstance(other, Value) else Value(other, label=f"{other:g}")
        out = Value(self.data - other.data, (self, other), "sub")

        def _backward():
            self.grad += out.grad
            other.grad += -out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad  # saturates → chokes the grad

        out._backward = _backward
        return out

    def topo(self):
        order, seen = [], set()

        def build(v):
            if id(v) not in seen:
                seen.add(id(v))
                for c in v._prev:
                    build(c)
                order.append(v)

        build(self)
        return order

    def backward(self):
        order = self.topo()
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


# ============================================================================
#  PLUMBING from here down — the net, the trainer, the debugger's tracer.
# ============================================================================

# Same PRNG as the JS side of the lab, so a given seed reproduces exactly.
def mulberry32(a):
    a &= 0xFFFFFFFF

    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (a | 1)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return r


SUB = "₀₁₂₃₄₅₆₇₈₉"

SEED = int(seed)
LR = float(lr)
E = int(epochs)
TRACE_EP = max(0, min(int(trace_epoch), E - 1))

# XOR in ±1 form: y = -x1·x2. Four points, one sample per iteration (SGD),
# so the whole computation graph stays small enough to read.
DATA = [([-1.0, -1.0], -1.0), ([-1.0, 1.0], 1.0), ([1.0, -1.0], 1.0), ([1.0, 1.0], -1.0)]

# ---- parameters (persistent Values — these are what learning changes) ------
_r = mulberry32(SEED)


def u():
    return _r() * 2.0 - 1.0


HID = 3
W1 = [[Value(u(), label=f"w{SUB[j+1]}₁"), Value(u(), label=f"w{SUB[j+1]}₂")] for j in range(HID)]
B1 = [Value(u(), label=f"b{SUB[j+1]}") for j in range(HID)]
W2 = [Value(u(), label=f"v{SUB[j+1]}") for j in range(HID)]
C = Value(u(), label="c")
PARAMS = [w for row in W1 for w in row] + B1 + W2 + [C]
PARAM_IDS = {id(p) for p in PARAMS}

# persistent input leaves — .data is swapped per sample, the objects survive
X1 = Value(0.0, label="x₁")
X2 = Value(0.0, label="x₂")
Y = Value(0.0, label="y")
LEAVES = [X1, X2, Y] + PARAMS


def forward():
    """Build the graph for one sample: 2 → HID(tanh) → 1(tanh), squared error."""
    hs = []
    for j in range(HID):
        m1 = X1 * W1[j][0]
        m1.label = f"x₁w{SUB[j+1]}₁"
        m2 = X2 * W1[j][1]
        m2.label = f"x₂w{SUB[j+1]}₂"
        s = m1 + m2
        s.label = f"Σ{SUB[j+1]}"
        z = s + B1[j]
        z.label = f"z{SUB[j+1]}"
        h = z.tanh()
        h.label = f"h{SUB[j+1]}"
        hs.append(h)
    tot = hs[0] * W2[0]
    tot.label = "h₁v₁"
    for j in range(1, HID):
        p = hs[j] * W2[j]
        p.label = f"h{SUB[j+1]}v{SUB[j+1]}"
        tot = tot + p
        tot.label = "Σ" if j == HID - 1 else f"Σ₁{SUB[j+1]}"
    zo = tot + C
    zo.label = "z"
    pred = zo.tanh()
    pred.label = "pred"
    d = pred - Y
    d.label = "d"
    # d * d, not a square() primitive: d occupies BOTH slots of one mul —
    # the canonical a·a moment where the += in __mul__._backward is what
    # makes ∂L/∂d come out as 2d instead of d
    L = d * d
    L.label = "L"
    return L


def kind_of(v):
    if v is X1 or v is X2:
        return "input"
    if v is Y:
        return "const"
    if id(v) in PARAM_IDS:
        return "param"
    # a leaf that is none of the above is a coerced literal like `s + 0.5`
    return "op" if v._prev else "const"


def build_ids(order):
    """Stable ids: leaves by fixed index, ops by topological position.
    The same code path builds the same graph shape every iteration, so a
    node's topo index — and therefore its id — is stable across samples."""
    ids = {}
    for i, v in enumerate(LEAVES):
        ids[id(v)] = f"n{i}"
    k = 0
    for v in order:
        if id(v) not in ids:
            ids[id(v)] = f"o{k}"
            k += 1
    return ids


def trace_iteration(si, events, want_structure):
    """One SGD step, narrated: emit an event per forward op, per chain-rule
    application, per weight nudge. The tracer only OBSERVES Value — if you
    break an op's _backward above, the events faithfully show the wrong grads."""
    L = forward()
    order = L.topo()
    ids = build_ids(order)

    structure = None
    if want_structure:
        edges, seen_e = [], set()
        for v in order:
            for c in v._prev:
                e = (ids[id(c)], ids[id(v)])
                if e not in seen_e:  # d·d would emit d→L twice
                    seen_e.add(e)
                    edges.append([e[0], e[1]])
        # loss subgraph = everything downstream of the target y; the UI
        # renders those forward events as the LOSS phase
        mark = {id(Y)}
        loss_ids = []
        for v in order:
            if v._prev and any(id(c) in mark for c in v._prev):
                mark.add(id(v))
                loss_ids.append(ids[id(v)])
        structure = {
            "nodes": [
                {"id": ids[id(v)], "kind": kind_of(v), "op": v._op, "label": v.label or v._op, "d0": round(v.data, 6)}
                for v in order
            ],
            "edges": edges,
            "loss": loss_ids,
        }

    events.append({"t": "sample", "si": si, "x": [X1.data, X2.data], "y": Y.data})
    for v in order:
        if v._prev:
            events.append({"t": "fwd", "id": ids[id(v)], "v": round(v.data, 6)})

    events.append({"t": "zero"})
    for lv in LEAVES:
        lv.grad = 0.0
    L.grad = 1.0
    events.append({"t": "seed", "id": ids[id(L)]})

    for v in reversed(order):
        if not v._prev:
            continue
        before = [c.grad for c in v._prev]
        v._backward()
        cs, seen_c = [], set()
        for c, b in zip(v._prev, before):
            cid = ids[id(c)]
            if cid in seen_c:
                continue  # duplicated child (d·d): one entry, delta = both writes
            seen_c.add(cid)
            cs.append({"id": cid, "d": round(c.grad - b, 6), "g": round(c.grad, 6)})
        events.append({"t": "bwd", "id": ids[id(v)], "g": round(v.grad, 6), "cs": cs})

    for p in PARAMS:
        old = p.data
        p.data -= LR * p.grad
        events.append(
            {"t": "upd", "id": ids[id(p)], "old": round(old, 6), "g": round(p.grad, 6), "new": round(p.data, 6)}
        )

    events.append({"t": "end", "k": si, "loss": round(L.data, 6)})
    return L.data, structure


def train_iteration():
    """The same step, silent — used for every epoch except the traced one."""
    L = forward()
    for lv in LEAVES:
        lv.grad = 0.0
    L.backward()
    for p in PARAMS:
        p.data -= LR * p.grad
    return L.data


# ---- numeric twin of forward(), for the decision surface + gradcheck -------
# (plumbing: pure floats, no Values — if you change the architecture above,
#  mirror it here)
def eval_net(a, b):
    z = C.data
    for j in range(HID):
        h = math.tanh(W1[j][0].data * a + W1[j][1].data * b + B1[j].data)
        z += h * W2[j].data
    return math.tanh(z)


R = 42
DOM = 1.8


def surface():
    g = []
    for row in range(R):  # row 0 = top of the plot (x2 = +DOM)
        yv = DOM - (row + 0.5) * 2.0 * DOM / R
        for col in range(R):
            xv = -DOM + (col + 0.5) * 2.0 * DOM / R
            g.append(round(eval_net(xv, yv), 4))
    return g


# ---- LIVE GRADIENT CHECK ----------------------------------------------------
# Does YOUR chain rule agree with finite differences? Checked at init over
# ALL 13 parameters — a single-weight check is blind to sabotage on branches
# it doesn't ride (e.g. delete other.grad in __add__ and only the b/c/h₂/h₃
# branches die). 26 eval_net calls; still instant.
X1.data, X2.data, Y.data = DATA[0][0][0], DATA[0][0][1], DATA[0][1]
_Lg = forward()
for _lv in LEAVES:
    _lv.grad = 0.0
_Lg.backward()  # <- analytic grads from YOUR _backward closures

_eps = 1e-6
gradcheck_err = 0.0
gradcheck_worst = ""
for _p in PARAMS:
    _o = _p.data
    _p.data = _o + _eps
    _lp = (eval_net(X1.data, X2.data) - Y.data) ** 2
    _p.data = _o - _eps
    _lm = (eval_net(X1.data, X2.data) - Y.data) ** 2
    _p.data = _o  # restore
    _numeric = (_lp - _lm) / (2.0 * _eps)
    _rel = abs(_p.grad - _numeric) / max(abs(_p.grad) + abs(_numeric), 1e-12)
    if _rel > gradcheck_err:
        gradcheck_err = _rel
        gradcheck_worst = _p.label
gradcheck_ok = bool(gradcheck_err < 1e-4)
gradcheck_err = float(gradcheck_err)

# ---- train, tracing the chosen epoch ----------------------------------------
events = []
structure = None
loss_hist = []
surfaces = []

for ep in range(E):
    tot = 0.0
    for si, (xv, yv) in enumerate(DATA):
        X1.data, X2.data, Y.data = xv[0], xv[1], yv
        if ep == TRACE_EP:
            if si == 0:
                surfaces.append(surface())  # before the epoch's first step
            lval, s = trace_iteration(si, events, want_structure=(si == 0))
            if s is not None:
                structure = s
            surfaces.append(surface())  # after each traced step
        else:
            lval = train_iteration()
        tot += lval
    loss_hist.append(round(tot / 4.0, 6))

final_loss = float(loss_hist[-1])
final_acc = sum(1 for (xv, yv) in DATA if (eval_net(xv[0], xv[1]) > 0) == (yv > 0)) / 4.0
print(
    f"seed={SEED} lr={LR} · {E} epochs · traced epoch {TRACE_EP} "
    f"({len(events)} events) · final loss={final_loss:.4f} acc={final_acc:.2f}"
)

# ---- outputs (JSON strings — no numpy needed for this module) ---------------
structure_json = json.dumps(structure)
trace_json = json.dumps(events)
surfaces_json = json.dumps(surfaces)
loss_json = json.dumps(loss_hist)
data_json = json.dumps([[x[0], x[1], y] for (x, y) in DATA])
traced_epoch = TRACE_EP
surf_r = R
surf_dom = DOM
lr_used = LR
# gradcheck_ok / gradcheck_err / gradcheck_worst are set above
