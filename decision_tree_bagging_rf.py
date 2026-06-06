import os
import numpy as np
from collections import Counter


# Step 1: load data
def load(fp="mnist.npz"):
    if not os.path.exists(fp):
        raise FileNotFoundError(
            f"File not found: {fp}\n"
            f"Current folder: {os.getcwd()}\n"
            f"Files here: {os.listdir()}"
        )
    d = np.load(fp)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


# Step 2: filter + normalize
def prep(xtr, ytr, xte, yte, cls=(0, 1, 2)):
    m1 = np.isin(ytr, cls)
    m2 = np.isin(yte, cls)

    xtr, ytr = xtr[m1], ytr[m1]
    xte, yte = xte[m2], yte[m2]

    xtr = xtr.reshape(len(xtr), -1).astype(float) / 255.0
    xte = xte.reshape(len(xte), -1).astype(float) / 255.0

    return xtr, ytr, xte, yte


# Step 3: PCA to reduced space
class PCA2:
    def __init__(self, k):
        self.k = k
        self.mu = None
        self.v = None

    def fit(self, X):
        self.mu = X.mean(axis=0)
        Xc = X - self.mu
        C = Xc.T @ Xc / Xc.shape[0]
        a, b = np.linalg.eigh(C)
        idx = np.argsort(a)[::-1]
        self.v = b[:, idx[:self.k]]
        return self

    def tr(self, X):
        return (X - self.mu) @ self.v

    def fit_tr(self, X):
        self.fit(X)
        return self.tr(X)


# Step 4: gini helpers
def cnt(y, cls):
    out = np.zeros(len(cls), dtype=int)
    for i, c in enumerate(cls):
        out[i] = np.sum(y == c)
    return out


def gini(y, cls):
    n = len(y)
    if n == 0:
        return 0.0
    p = cnt(y, cls) / n
    return 1.0 - np.sum(p ** 2)


def wg(y1, y2, cls):
    n = len(y1) + len(y2)
    if n == 0:
        return 0.0
    return (len(y1) / n) * gini(y1, cls) + (len(y2) / n) * gini(y2, cls)


def maj(y):
    if len(y) == 0:
        return None
    c = Counter(y)
    return c.most_common(1)[0][0]


def acc(y, yp):
    return np.mean(y == yp)


def cacc(y, yp, cls):
    out = {}
    for c in cls:
        m = (y == c)
        out[c] = np.mean(yp[m] == y[m]) if np.sum(m) > 0 else 0.0
    return out


# Step 5: choose threshold
def thr(x, mode="median"):
    if mode == "median":
        return np.median(x)
    if mode == "mean":
        return np.mean(x)
    raise ValueError("mode must be 'median' or 'mean'")


# Step 6: best split on all p dims
def best_split(X, y, cls, dims=None, mode="median"):
    p = X.shape[1]
    if dims is None:
        dims = np.arange(p)

    best = {
        "j": None,
        "t": None,
        "g": np.inf,
        "nl": 0,
        "nr": 0,
        "gl": 0.0,
        "gr": 0.0
    }

    for j in dims:
        t = thr(X[:, j], mode)
        ml = X[:, j] <= t
        mr = ~ml

        yl = y[ml]
        yr = y[mr]

        gv = wg(yl, yr, cls)

        if gv < best["g"]:
            best["j"] = int(j)
            best["t"] = float(t)
            best["g"] = float(gv)
            best["nl"] = int(len(yl))
            best["nr"] = int(len(yr))
            best["gl"] = float(gini(yl, cls))
            best["gr"] = float(gini(yr, cls))

    return best


# Step 7: build 2-level tree with 3 leaves
def fit_tree2(X, y, cls, first_j=None, mode="median"):
    if first_j is None:
        r1 = best_split(X, y, cls, dims=np.arange(X.shape[1]), mode=mode)
    else:
        r1 = best_split(X, y, cls, dims=[first_j], mode=mode)

    j1, t1 = r1["j"], r1["t"]
    m1 = X[:, j1] <= t1
    m2 = ~m1

    X1, y1 = X[m1], y[m1]
    X2, y2 = X[m2], y[m2]

    b1 = best_split(X1, y1, cls, dims=np.arange(X.shape[1]), mode=mode) if len(y1) > 1 else None
    b2 = best_split(X2, y2, cls, dims=np.arange(X.shape[1]), mode=mode) if len(y2) > 1 else None

    g1 = b1["g"] if b1 is not None else np.inf
    g2 = b2["g"] if b2 is not None else np.inf

    tr = {
        "j1": j1,
        "t1": t1,
        "split_side": None,
        "j2": None,
        "t2": None,
        "lab": {}
    }

    if g1 <= g2:
        tr["split_side"] = "L"
        tr["j2"] = b1["j"]
        tr["t2"] = b1["t"]

        m11 = X1[:, tr["j2"]] <= tr["t2"]
        m12 = ~m11

        y11 = y1[m11]
        y12 = y1[m12]

        tr["lab"]["LL"] = maj(y11)
        tr["lab"]["LR"] = maj(y12)
        tr["lab"]["R"] = maj(y2)

        tr["cnt"] = {
            "LL": len(y11),
            "LR": len(y12),
            "R": len(y2)
        }

    else:
        tr["split_side"] = "R"
        tr["j2"] = b2["j"]
        tr["t2"] = b2["t"]

        m21 = X2[:, tr["j2"]] <= tr["t2"]
        m22 = ~m21

        y21 = y2[m21]
        y22 = y2[m22]

        tr["lab"]["L"] = maj(y1)
        tr["lab"]["RL"] = maj(y21)
        tr["lab"]["RR"] = maj(y22)

        tr["cnt"] = {
            "L": len(y1),
            "RL": len(y21),
            "RR": len(y22)
        }

    tr["r1"] = r1
    tr["b1"] = b1
    tr["b2"] = b2
    return tr


# Step 8: tree prediction
def pred_one(x, tr):
    if x[tr["j1"]] <= tr["t1"]:
        if tr["split_side"] == "L":
            if x[tr["j2"]] <= tr["t2"]:
                return tr["lab"]["LL"]
            return tr["lab"]["LR"]
        return tr["lab"]["L"]
    else:
        if tr["split_side"] == "R":
            if x[tr["j2"]] <= tr["t2"]:
                return tr["lab"]["RL"]
            return tr["lab"]["RR"]
        return tr["lab"]["R"]


def pred_tree(X, tr):
    yp = np.empty(len(X), dtype=int)
    for i in range(len(X)):
        yp[i] = pred_one(X[i], tr)
    return yp


# Step 9: bootstrap sample
def boot(X, y, sd=None):
    rng = np.random.default_rng(sd)
    n = len(X)
    ib = rng.integers(0, n, size=n)
    oo = np.setdiff1d(np.arange(n), np.unique(ib))
    return X[ib], y[ib], ib, oo


# Step 10: bagging with oob
def fit_bag(X, y, cls, b=5, mode="median", sd=42):
    arr = []
    oob = []

    for i in range(b):
        xb, yb, ib, oo = boot(X, y, sd + i)
        tr = fit_tree2(xb, yb, cls, first_j=None, mode=mode)
        arr.append(tr)
        oob.append(oo)

    return arr, oob


def pred_many(X, arr):
    pr = []
    for tr in arr:
        pr.append(pred_tree(X, tr))
    return np.array(pr)


def vote(P, cls):
    out = np.empty(P.shape[1], dtype=int)
    for i in range(P.shape[1]):
        c = Counter(P[:, i])
        out[i] = c.most_common(1)[0][0]
    return out


def oob_err(X, y, arr, oob):
    n = len(X)
    box = [[] for _ in range(n)]

    for i, tr in enumerate(arr):
        oo = oob[i]
        if len(oo) == 0:
            continue
        yp = pred_tree(X[oo], tr)
        for j, idv in enumerate(oo):
            box[idv].append(yp[j])

    yp = np.full(n, -1, dtype=int)
    use = np.zeros(n, dtype=bool)

    for i in range(n):
        if len(box[i]) > 0:
            c = Counter(box[i])
            yp[i] = c.most_common(1)[0][0]
            use[i] = True

    if np.sum(use) == 0:
        return None, 0.0

    er = 1.0 - np.mean(yp[use] == y[use])
    return yp, er


# Step 11: random forest
def fit_tree_rf(X, y, cls, k=3, mode="median", sd=0):
    rng = np.random.default_rng(sd)
    p = X.shape[1]

    d1 = rng.choice(p, size=min(k, p), replace=False)
    r1 = best_split(X, y, cls, dims=d1, mode=mode)

    j1, t1 = r1["j"], r1["t"]
    m1 = X[:, j1] <= t1
    m2 = ~m1

    X1, y1 = X[m1], y[m1]
    X2, y2 = X[m2], y[m2]

    d2l = rng.choice(p, size=min(k, p), replace=False)
    d2r = rng.choice(p, size=min(k, p), replace=False)

    b1 = best_split(X1, y1, cls, dims=d2l, mode=mode) if len(y1) > 1 else None
    b2 = best_split(X2, y2, cls, dims=d2r, mode=mode) if len(y2) > 1 else None

    g1 = b1["g"] if b1 is not None else np.inf
    g2 = b2["g"] if b2 is not None else np.inf

    tr = {
        "j1": j1,
        "t1": t1,
        "split_side": None,
        "j2": None,
        "t2": None,
        "lab": {}
    }

    if g1 <= g2:
        tr["split_side"] = "L"
        tr["j2"] = b1["j"]
        tr["t2"] = b1["t"]

        m11 = X1[:, tr["j2"]] <= tr["t2"]
        m12 = ~m11

        y11 = y1[m11]
        y12 = y1[m12]

        tr["lab"]["LL"] = maj(y11)
        tr["lab"]["LR"] = maj(y12)
        tr["lab"]["R"] = maj(y2)

    else:
        tr["split_side"] = "R"
        tr["j2"] = b2["j"]
        tr["t2"] = b2["t"]

        m21 = X2[:, tr["j2"]] <= tr["t2"]
        m22 = ~m21

        y21 = y2[m21]
        y22 = y2[m22]

        tr["lab"]["L"] = maj(y1)
        tr["lab"]["RL"] = maj(y21)
        tr["lab"]["RR"] = maj(y22)

    tr["r1"] = r1
    tr["b1"] = b1
    tr["b2"] = b2
    return tr


def fit_rf(X, y, cls, b=5, k=3, mode="median", sd=42):
    arr = []
    oob = []

    for i in range(b):
        xb, yb, ib, oo = boot(X, y, sd + i)
        tr = fit_tree_rf(xb, yb, cls, k=k, mode=mode, sd=sd + 100 + i)
        arr.append(tr)
        oob.append(oo)

    return arr, oob


# Step 12: full run
def main():
    cls = [0, 1, 2]

    print("Step 1: loading data")
    xr, yr, xs, ys = load("mnist.npz")
    print("raw train:", xr.shape, yr.shape)
    print("raw test :", xs.shape, ys.shape)
    print()

    print("Step 2: filtering classes and flattening")
    xtr, ytr, xte, yte = prep(xr, yr, xs, ys, cls)
    print("filtered train:", xtr.shape, ytr.shape)
    print("filtered test :", xte.shape, yte.shape)
    print("classes train:", np.unique(ytr))
    print("classes test :", np.unique(yte))
    print()

    print("Step 3: PCA with p = 10")
    p = PCA2(10)
    ztr = p.fit_tr(xtr)
    zte = p.tr(xte)
    print("ztr shape:", ztr.shape)
    print("zte shape:", zte.shape)
    print("PCA matrix:", p.v.shape)
    print("Reduced to 10 dims?", ztr.shape[1] == 10 and zte.shape[1] == 10)
    print()

    print("Step 4: root split candidates on all 10 dims")
    vals = []
    for j in range(ztr.shape[1]):
        t = thr(ztr[:, j], mode="median")
        ml = ztr[:, j] <= t
        mr = ~ml
        gv = wg(ytr[ml], ytr[mr], cls)
        vals.append((j, t, gv, np.sum(ml), np.sum(mr)))
        print(f"dim={j} | thr={t:.6f} | wgini={gv:.6f} | left={np.sum(ml)} | right={np.sum(mr)}")
    print()

    print("Step 5: train tree with 3 leaves")
    tr = fit_tree2(ztr, ytr, cls, first_j=None, mode="median")
    print("root dim:", tr["j1"])
    print("root thr:", tr["t1"])
    print("root wgini:", tr["r1"]["g"])
    print("second split side:", tr["split_side"])
    print("second dim:", tr["j2"])
    print("second thr:", tr["t2"])
    print("leaf counts:", tr["cnt"])
    print("leaf labels:", tr["lab"])
    print()

    print("Step 6: predict on test set with decision tree")
    yp = pred_tree(zte, tr)
    a = acc(yte, yp)
    ca = cacc(yte, yp, cls)
    print("overall test accuracy:", a)
    print("overall test accuracy %:", round(a * 100, 2))
    print("class-wise accuracy:", ca)
    print()

    print("Step 7: bagging with 5 bootstraps")
    bag, oob_b = fit_bag(ztr, ytr, cls, b=5, mode="median", sd=42)

    for i in range(len(bag)):
        print(
            f"tree={i+1} | root dim={bag[i]['j1']} | root thr={bag[i]['t1']:.6f} | "
            f"oob count={len(oob_b[i])}"
        )
    print()

    print("Step 8: OOB error for bagging")
    yp_oob_b, er_b = oob_err(ztr, ytr, bag, oob_b)
    print("bagging average OOB error:", er_b)
    print("bagging average OOB accuracy:", 1 - er_b if yp_oob_b is not None else None)
    print()

    print("Step 9: bagging test voting")
    P_b = pred_many(zte, bag)
    yb = vote(P_b, cls)
    ab = acc(yte, yb)
    cab = cacc(yte, yb, cls)
    print("bagging test accuracy:", ab)
    print("bagging test accuracy %:", round(ab * 100, 2))
    print("bagging class-wise accuracy:", cab)
    print("prediction matrix shape:", P_b.shape)
    print()

    print("Step 10: random forest with 5 trees")
    k = int(np.sqrt(ztr.shape[1]))
    rf, oob_r = fit_rf(ztr, ytr, cls, b=5, k=k, mode="median", sd=99)
    print("chosen k:", k)
    for i in range(len(rf)):
        print(
            f"tree={i+1} | root dim={rf[i]['j1']} | root thr={rf[i]['t1']:.6f} | "
            f"oob count={len(oob_r[i])}"
        )
    print()

    print("Step 11: OOB error for random forest")
    yp_oob_r, er_r = oob_err(ztr, ytr, rf, oob_r)
    print("rf OOB error:", er_r)
    print("rf OOB accuracy:", 1 - er_r if yp_oob_r is not None else None)
    print()

    print("Step 12: random forest test voting")
    P_r = pred_many(zte, rf)
    yrf = vote(P_r, cls)
    arf = acc(yte, yrf)
    carf = cacc(yte, yrf, cls)
    print("rf test accuracy:", arf)
    print("rf test accuracy %:", round(arf * 100, 2))
    print("rf class-wise accuracy:", carf)
    print("prediction matrix shape:", P_r.shape)
    print()

    print("Step 13: final comparison")
    print("single tree test acc :", round(a * 100, 2))
    print("bagging test acc     :", round(ab * 100, 2))
    print("random forest acc    :", round(arf * 100, 2))
    print("bagging OOB err      :", round(er_b, 6))
    print("random forest OOB err:", round(er_r, 6))

    if arf > ab:
        print("Random Forest is better here because feature randomness reduces tree correlation.")
    elif arf < ab:
        print("Bagging is better here on this run; random feature restriction may have removed useful splits.")
    else:
        print("Both methods are tied on this run.")


if __name__ == "__main__":
    main()