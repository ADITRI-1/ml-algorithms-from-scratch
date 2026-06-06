# Decision Trees, Bagging, Random Forest & Regularization from Scratch

A from-scratch implementation of tree-based ensemble methods and regularized regression on MNIST and Fashion-MNIST, covering Decision Trees, Bagging, Random Forest, Ridge Regression, and Lasso. Core algorithms are built using NumPy only — scikit-learn is used only for Lasso.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Algorithms](#algorithms)
- [Results](#results)
- [Key Design Choices](#key-design-choices)

---

## Overview

This repo covers three experiments:

| Experiment | Dataset | Task | Method |
|---|---|---|---|
| Trees & Ensembles | MNIST (digits 0, 1, 2) | Multi-class classification | Decision Tree (3 leaves) → Bagging → Random Forest |
| Regression Bagging | Fashion-MNIST (classes 0, 1, 2) | Regression | Regression Stump → Bagging |
| Regularized Regression | MNIST (digits 0, 1, 2) | Multi-class classification | Ridge & Lasso (one-vs-rest) |

All experiments reduce 784-d pixel vectors to 10 PCA dimensions before fitting any model.

---

## Dataset

### MNIST (Trees & Regularization)
- **File:** `mnist.npz`
- **Filtered classes:** Digits `0`, `1`, `2`
- **Preprocessing:** Flatten 28×28 → 784-d, normalize to `[0, 1]`, PCA → 10-d

| Split | Samples |
|---|---|
| Train | 18,623 |
| Test | 3,147 |

### Fashion-MNIST (Regression Bagging)
- **Source:** `tensorflow.keras.datasets.fashion_mnist`
- **Filtered classes:** `0` (T-shirt), `1` (Trouser), `2` (Pullover)
- **Preprocessing:** Same as MNIST — flatten, normalize, PCA → 10-d

| Split | Samples |
|---|---|
| Train | 18,000 |
| Test | 3,000 |

> Download `mnist.npz` from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) or save via `np.savez`. Fashion-MNIST downloads automatically via Keras.

---

## Requirements

Python 3.8+ recommended.

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | All core computation |
| `matplotlib` | ≥ 3.4 | Plots |
| `scikit-learn` | ≥ 0.24 | Lasso solver only (`sklearn.linear_model.Lasso`) |
| `tensorflow` | ≥ 2.x | Fashion-MNIST loader only |

> Decision Trees, Bagging, Random Forest, PCA, Ridge Regression, and all evaluation metrics are implemented from scratch using NumPy.

---
---

## Algorithms

### Decision Tree / Bagging / Random Forest (`trees.py`)

**Decision Tree (3 leaves — 2-level)**
- Split criterion: Gini impurity (weighted)
- Threshold strategy: median of feature values
- Structure: one root split → second split on the better child → 3 leaf nodes
- Leaf prediction: majority class vote

**Bagging**
- 5 bootstrap samples (with replacement, ~63% unique samples per bag)
- Each tree is a full 2-level decision tree on all 10 PCA dims
- Final prediction: majority vote across all 5 trees
- OOB error estimated from out-of-bag samples per tree

**Random Forest**
- Same bootstrap procedure as bagging
- Feature subsampling: `k = floor(sqrt(p)) = 3` features randomly selected at each node
- Separate random feature subsets drawn independently at root and child nodes

---

### Regression Stump + Bagging (`regression.py`)

- **Target:** Raw class labels `{0, 1, 2}` as continuous regression targets
- **Weak learner:** Single regression stump — exhaustive search over all midpoint thresholds on all 10 PCA dims, minimizing SSR (Sum of Squared Residuals)
- **Leaf value:** Mean of training targets in each partition
- **Bagging:** 5 stumps trained on independent bootstrap samples; final prediction is the average of all 5 stump outputs
- **OOB MSE:** Computed independently per stump on its out-of-bag samples, then averaged

---

### Ridge & Lasso Regression (`regularized.py`)

- **Formulation:** One-vs-rest — 3 binary regressors trained simultaneously, one per class
- **Ridge:** Closed-form solution — `w = (XᵀX + λI)⁻¹ Xᵀy` (bias term unregularized)
- **Lasso:** `sklearn.linear_model.Lasso` per class, `max_iter=10000`
- **Lambda sweep:** `λ ∈ {1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100}`
- **Classification:** Argmax over the 3 regression scores
- **Model complexity study:** Ridge re-run for PCA dims `p ∈ {2, 5, 10, 20, 30}`

---

## Results

### Classification Accuracy (MNIST digits 0/1/2) — Trees

| Method | Test Accuracy | OOB Error |
|---|---|---|
| Single Decision Tree (3 leaves) | 78.17% | — |
| Bagging (5 trees) | 78.30% | 0.2128 |
| **Random Forest (5 trees, k=3)** | **80.20%** ✅ | 0.3646 |

Class-wise accuracy (Random Forest): `0 → 61.4%`, `1 → 85.9%`, `2 → 91.8%`

Random Forest outperforms bagging due to feature randomness reducing tree correlation, despite a higher OOB error (OOB is estimated on a limited 2-level tree, so it is a noisier estimate here).

---

### Regression MSE (Fashion-MNIST classes 0/1/2) — Bagging

| Method | Test MSE | Avg OOB MSE |
|---|---|---|
| Single Regression Stump | 0.3629 | — |
| **Bagging (5 stumps)** | **0.3578** ✅ | 0.3472 |

Bagging reduces test MSE by ~0.005 over a single stump by averaging out variance across bootstrap samples. All 5 stumps independently selected PCA dim 2 as the best split feature.

---

### Classification Accuracy (MNIST digits 0/1/2) — Ridge vs Lasso

**Lambda sweep (val MSE, p=10 PCA dims):**

| λ | Ridge Val MSE | Lasso Val MSE | Lasso Non-zero Coefs |
|---|---|---|---|
| 1e-4 | 0.050240 | 0.050240 | 30 |
| 1e-3 | 0.050240 | 0.050242 | 30 |
| 1e-2 | 0.050240 | 0.050682 | 29 |
| 0.1 | 0.050240 | 0.064702 | 10 |
| 1 | 0.050240 | 0.185579 | 2 |
| 10 | 0.050240 | 0.221530 | 0 |
| 100 | **0.050238** ✅ | 0.221530 | 0 |

**Best λ:** Ridge → `100`, Lasso → `1e-4`

**Model complexity (Ridge, best λ=100):**

| PCA dims | Train MSE | Val MSE |
|---|---|---|
| 2 | 0.067952 | 0.068548 |
| 5 | 0.053726 | 0.053992 |
| 10 | 0.049973 | 0.050238 |
| 20 | 0.041641 | 0.041922 |
| 30 | 0.039224 | 0.039582 |

**Final test accuracy (p=10):**

| Method | Test Accuracy |
|---|---|
| Ridge (λ=100) | **94.76%** |
| Lasso (λ=1e-4) | **94.76%** |

Both achieve identical test accuracy. Ridge's optimal λ is large (heavy shrinkage) because the 10 PCA features are already well-conditioned. Lasso at λ=1e-4 retains all 30 coefficients (10 features × 3 classes) and matches Ridge exactly.

---

## Key Design Choices

- **PCA fitted on train only** — val and test sets projected using training mean and eigenvectors to prevent data leakage.
- **Median threshold for trees** — one split point per feature per node; avoids exhaustive threshold search while remaining deterministic.
- **2-level tree with 3 leaves** — the second split is chosen on the child node with lower weighted Gini, not both children, to strictly produce 3 leaf nodes.
- **OOB estimation** — each sample's OOB prediction is aggregated across only the trees that did not see it during bootstrap, giving an unbiased generalization estimate without a separate validation set.
- **SSR stump for regression** — exhaustive midpoint search across all features minimizes SSR directly; leaf values are the mean of targets in each partition, which is the optimal L2 predictor.
- **Unregularized bias in Ridge** — the identity matrix used in the closed-form solution has `I[0,0] = 0` so the intercept term is not penalized.
- **One-vs-rest argmax classification** — regression scores are not calibrated probabilities; the class with the highest raw score is assigned, which works well when classes are well-separated in PCA space.
