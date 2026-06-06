import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist


# Step 1: load Fashion-MNIST
def load():
    (xtr, ytr), (xte, yte) = fashion_mnist.load_data()
    return xtr, ytr, xte, yte


# Step 2: filter + normalize
def prep(xtr, ytr, xte, yte, cls=(0, 1, 2)):
    m1 = np.isin(ytr, cls)
    m2 = np.isin(yte, cls)

    xtr, ytr = xtr[m1], ytr[m1]
    xte, yte = xte[m2], yte[m2]

    xtr = xtr.reshape(len(xtr), -1).astype(float) / 255.0
    xte = xte.reshape(len(xte), -1).astype(float) / 255.0

    return xtr, ytr, xte, yte


# Step 3: PCA to p = 10
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


# Step 4: targets for regression
def yy(y):
    return y.astype(float)


# Step 5: mse + ssr helpers
def mse(y, yp):
    return np.mean((y - yp) ** 2)


def ssr(y1, y2):
    s = 0.0
    if len(y1) > 0:
        m1 = np.mean(y1)
        s += np.sum((y1 - m1) ** 2)
    if len(y2) > 0:
        m2 = np.mean(y2)
        s += np.sum((y2 - m2) ** 2)
    return s


# Step 6: threshold candidates = midpoints
def mids(x):
    u = np.unique(np.sort(x))
    if len(u) < 2:
        return np.array([])
    return (u[:-1] + u[1:]) / 2.0


# Step 7: best regression stump
def stump(X, y):
    p = X.shape[1]
    best = {
        "j": None,
        "t": None,
        "s": np.inf,
        "ml": None,
        "mr": None,
        "yl": None,
        "yr": None
    }

    for j in range(p):
        ts = mids(X[:, j])
        if len(ts) == 0:
            continue

        for t in ts:
            ml = X[:, j] <= t
            mr = ~ml

            if np.sum(ml) == 0 or np.sum(mr) == 0:
                continue

            yl = y[ml]
            yr = y[mr]
            sv = ssr(yl, yr)

            if sv < best["s"]:
                best["j"] = int(j)
                best["t"] = float(t)
                best["s"] = float(sv)
                best["ml"] = ml
                best["mr"] = mr
                best["yl"] = float(np.mean(yl))
                best["yr"] = float(np.mean(yr))

    return best


# Step 8: stump prediction
def pred_stump(X, st):
    yp = np.empty(len(X), dtype=float)
    m = X[:, st["j"]] <= st["t"]
    yp[m] = st["yl"]
    yp[~m] = st["yr"]
    return yp


# Step 9: bootstrap
def boot(X, y, sd=None):
    rng = np.random.default_rng(sd)
    n = len(X)
    ib = rng.integers(0, n, size=n)
    oo = np.setdiff1d(np.arange(n), np.unique(ib))
    return X[ib], y[ib], ib, oo


# Step 10: bagged stumps
def fit_bag(X, y, b=5, sd=42):
    arr = []
    oob = []

    for i in range(b):
        xb, yb, ib, oo = boot(X, y, sd + i)
        st = stump(xb, yb)
        arr.append(st)
        oob.append(oo)

    return arr, oob


def pred_many(X, arr):
    P = []
    for st in arr:
        P.append(pred_stump(X, st))
    return np.array(P)


def pred_avg(X, arr):
    P = pred_many(X, arr)
    return np.mean(P, axis=0)


# Step 11: oob mse
def oob_mse(X, y, arr, oob):
    n = len(X)
    box = [[] for _ in range(n)]

    for i, st in enumerate(arr):
        oo = oob[i]
        if len(oo) == 0:
            continue
        yp = pred_stump(X[oo], st)
        for j, idx in enumerate(oo):
            box[idx].append(yp[j])

    yp = np.full(n, np.nan)
    use = np.zeros(n, dtype=bool)

    for i in range(n):
        if len(box[i]) > 0:
            yp[i] = np.mean(box[i])
            use[i] = True

    if np.sum(use) == 0:
        return yp, None

    return yp, mse(y[use], yp[use])


# Step 12: full run
def main():
    cls = [0, 1, 2]

    print("Step 1: loading Fashion-MNIST")
    xr, yr, xs, ys = load()
    print("raw train:", xr.shape, yr.shape)
    print("raw test :", xs.shape, ys.shape)
    print()

    print("Step 2: filtering classes and flattening")
    xtr, ytr, xte, yte = prep(xr, yr, xs, ys, cls)
    print("filtered train:", xtr.shape, ytr.shape)
    print("filtered test :", xte.shape, yte.shape)
    print("classes in train:", np.unique(ytr))
    print("classes in test :", np.unique(yte))
    print()

    print("Step 3: PCA with p = 10")
    p = PCA2(10)
    ztr = p.fit_tr(xtr)
    zte = p.tr(xte)
    print("ztr shape:", ztr.shape)
    print("zte shape:", zte.shape)
    print("PCA matrix shape:", p.v.shape)
    print("Reduced to 10 dims?", ztr.shape[1] == 10 and zte.shape[1] == 10)
    print()

    print("Step 4: regression targets")
    yt = yy(ytr)
    yv = yy(yte)
    print("yt shape:", yt.shape)
    print("yv shape:", yv.shape)
    print("target values in train:", np.unique(yt))
    print()

    print("Step 5: fit single regression stump")
    st = stump(ztr, yt)
    print("best dim:", st["j"])
    print("best thr:", st["t"])
    print("best SSR:", st["s"])
    print("left mean:", st["yl"])
    print("right mean:", st["yr"])
    print("left count:", int(np.sum(st["ml"])))
    print("right count:", int(np.sum(st["mr"])))
    print()

    print("Step 6: test prediction for single stump")
    yp1 = pred_stump(zte, st)
    m1 = mse(yv, yp1)
    print("single stump test MSE:", m1)
    print("prediction shape:", yp1.shape)
    print()

    print("Step 7: bagging with 5 bootstraps")
    bag, oob = fit_bag(ztr, yt, b=5, sd=42)
    for i, st in enumerate(bag):
        print(
            f"stump={i+1} | dim={st['j']} | thr={st['t']:.6f} | "
            f"SSR={st['s']:.6f} | oob count={len(oob[i])}"
        )
    print()

    print("Step 8: OOB MSE for bagging")
    yp_oob, moob = oob_mse(ztr, yt, bag, oob)
    print("bagging OOB MSE:", moob)
    print()

    print("Step 9: bagging test prediction")
    yp2 = pred_avg(zte, bag)
    m2 = mse(yv, yp2)
    print("bagging test MSE:", m2)
    print("prediction shape:", yp2.shape)
    print()

    print("Step 10: compare single stump vs bagging")
    print("single stump test MSE:", m1)
    print("bagging test MSE     :", m2)
    print("bagging OOB MSE      :", moob)
    if m2 < m1:
        print("Bagging is better here because averaging reduces variance.")
    elif m2 > m1:
        print("Single stump is better on this run; bagging did not help enough.")
    else:
        print("Both methods are tied on this run.")
    print()

    print("Step 11: plot predictions on same figure")
    n = min(300, len(yv))
    ix = np.arange(n)

    plt.figure(figsize=(12, 5))
    plt.plot(ix, yv[:n], label="true", linewidth=2)
    plt.plot(ix, yp1[:n], label="stump", linestyle="--")
    plt.plot(ix, yp2[:n], label="bagging", linestyle=":")
    plt.xlabel("sample index")
    plt.ylabel("response")
    plt.title("Single stump vs Bagging on test samples")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()