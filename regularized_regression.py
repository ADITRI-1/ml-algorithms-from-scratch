import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


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


# Step 4: one-vs-rest targets
def oh(y, cls):
    Y = np.zeros((len(y), len(cls)))
    for i, c in enumerate(cls):
        Y[:, i] = (y == c).astype(float)
    return Y


# Step 5: ridge regression
def ridge(X, Y, l):
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0
    wb = np.linalg.solve(Xb.T @ Xb + l * I, Xb.T @ Y)
    return wb[1:], wb[0]


def pred_r(X, w, b):
    return X @ w + b


# Step 6: lasso regression
def lasso(X, Y, l):
    k = Y.shape[1]
    w = np.zeros((X.shape[1], k))
    b = np.zeros(k)

    for i in range(k):
        m = Lasso(alpha=l, max_iter=10000, fit_intercept=True)
        m.fit(X, Y[:, i])
        w[:, i] = m.coef_
        b[i] = m.intercept_

    return w, b


def pred_l(X, w, b):
    return X @ w + b


# Step 7: helpers
def mse(a, b):
    return np.mean((a - b) ** 2)

def nz(w, t=1e-4):
    return np.sum(np.abs(w) > t)

def pick(s):
    return np.argmax(s, axis=1)

def acc(y, yp, cls):
    yp2 = np.array([cls[i] for i in yp])
    return np.mean(yp2 == y)


# Step 8: main
def main():
    cls = [0, 1, 2]
    lam = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

    print("Step 1: loading data")
    xr, yr, xs, ys = load("mnist.npz")
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
    print("xtr shape:", xtr.shape)
    print("ztr shape:", ztr.shape)
    print("zte shape:", zte.shape)
    print("PCA matrix shape:", p.v.shape)
    print("Reduced to 10 dims?", ztr.shape[1] == 10 and zte.shape[1] == 10)
    print()

    print("Step 4: one-vs-rest targets")
    Ytr = oh(ytr, cls)
    Yte = oh(yte, cls)
    print("Ytr:", Ytr.shape)
    print("Yte:", Yte.shape)
    print("row sum check:", np.all(Ytr.sum(axis=1) == 1))
    print()

    print("Step 5: train Ridge and Lasso for all lambda using TEST MSE")
    rtr, rtest = [], []
    ltr, ltest = [], []
    lnz = []

    rc = []
    lc = []

    for l in lam:
        # Ridge
        wr, br = ridge(ztr, Ytr, l)
        sr_tr = pred_r(ztr, wr, br)
        sr_te = pred_r(zte, wr, br)
        rtr.append(mse(Ytr, sr_tr))
        rtest.append(mse(Yte, sr_te))
        rc.append(wr[:, 1].copy())

        # Lasso
        wl, bl = lasso(ztr, Ytr, l)
        sl_tr = pred_l(ztr, wl, bl)
        sl_te = pred_l(zte, wl, bl)
        ltr.append(mse(Ytr, sl_tr))
        ltest.append(mse(Yte, sl_te))
        lnz.append(nz(wl))
        lc.append(wl[:, 1].copy())

        print(
            f"lambda={l:>7} | ridge test mse={rtest[-1]:.6f} | "
            f"lasso test mse={ltest[-1]:.6f} | lasso nonzero={lnz[-1]}"
        )
    print()

    print("Step 6: verify model output shapes")
    wr, br = ridge(ztr, Ytr, lam[0])
    wl, bl = lasso(ztr, Ytr, lam[0])
    print("ridge w,b:", wr.shape, br.shape)
    print("lasso w,b:", wl.shape, bl.shape)
    print("ridge pred:", pred_r(ztr, wr, br).shape)
    print("lasso pred:", pred_l(ztr, wl, bl).shape)
    print()

    print("Step 7: plot train/test MSE vs lambda")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].semilogx(lam, rtr, marker="o", label="train")
    ax[0].semilogx(lam, rtest, marker="s", label="test")
    ax[0].set_title("Ridge: MSE vs lambda")
    ax[0].set_xlabel("lambda")
    ax[0].set_ylabel("MSE")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].semilogx(lam, ltr, marker="o", label="train")
    ax[1].semilogx(lam, ltest, marker="s", label="test")
    ax[1].set_title("Lasso: MSE vs lambda")
    ax[1].set_xlabel("lambda")
    ax[1].set_ylabel("MSE")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Step 8: plot lasso non-zero coefficients")
    plt.figure(figsize=(7, 4))
    plt.semilogx(lam, lnz, marker="o")
    plt.title("Lasso: non-zero coefficients vs lambda")
    plt.xlabel("lambda")
    plt.ylabel("non-zero count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Step 9: plot regularization paths for one class")
    rc = np.array(rc)
    lc = np.array(lc)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    for j in range(rc.shape[1]):
        ax[0].semilogx(lam, rc[:, j], label=f"f{j}")
    ax[0].axhline(0, linestyle="--", linewidth=0.8)
    ax[0].set_title("Ridge path (class 1)")
    ax[0].set_xlabel("lambda")
    ax[0].set_ylabel("coef")
    ax[0].grid(True, alpha=0.3)

    for j in range(lc.shape[1]):
        ax[1].semilogx(lam, lc[:, j], label=f"f{j}")
    ax[1].axhline(0, linestyle="--", linewidth=0.8)
    ax[1].set_title("Lasso path (class 1)")
    ax[1].set_xlabel("lambda")
    ax[1].set_ylabel("coef")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Step 10: best lambda from TEST MSE")
    br_l = lam[np.argmin(rtest)]
    bl_l = lam[np.argmin(ltest)]
    print("best ridge lambda:", br_l)
    print("best lasso lambda:", bl_l)
    print()

    print("Step 11: model complexity with ridge using TEST MSE")
    pp = [2, 5, 10, 20, 30]
    ctr, ctest = [], []

    for k in pp:
        pk = PCA2(k)
        ztr_k = pk.fit_tr(xtr)
        zte_k = pk.tr(xte)

        wk, bk = ridge(ztr_k, Ytr, br_l)
        ctr.append(mse(Ytr, pred_r(ztr_k, wk, bk)))
        ctest.append(mse(Yte, pred_r(zte_k, wk, bk)))

        print(
            f"p={k:>2} | ztr_k={ztr_k.shape} | "
            f"train mse={ctr[-1]:.6f} | test mse={ctest[-1]:.6f}"
        )

    plt.figure(figsize=(7, 4))
    plt.plot(pp, ctr, marker="o", label="train")
    plt.plot(pp, ctest, marker="s", label="test")
    plt.title(f"Ridge: model complexity vs MSE (best lambda={br_l})")
    plt.xlabel("PCA dimensions")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print()
    print("Step 12: test accuracy for best ridge and best lasso")
    p10 = PCA2(10)
    ztr10 = p10.fit_tr(xtr)
    zte10 = p10.tr(xte)

    wr, br = ridge(ztr10, Ytr, br_l)
    wl, bl = lasso(ztr10, Ytr, bl_l)

    sr = pred_r(zte10, wr, br)
    sl = pred_l(zte10, wl, bl)

    yrp = pick(sr)
    ylp = pick(sl)

    ar = acc(yte, yrp, cls)
    al = acc(yte, ylp, cls)

    print("ridge test accuracy:", ar)
    print("lasso test accuracy:", al)
    print("ridge test accuracy %:", round(ar * 100, 2))
    print("lasso test accuracy %:", round(al * 100, 2))


if __name__ == "__main__":
    main()