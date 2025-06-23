import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, qr
from sklearn.model_selection import train_test_split

# ----------------------
# 1. Data Generation (Hierarchical Data)
# ----------------------
clusters_per_class = {1: [300], 2: [150,150], 3: [75,75,75,75]}
dims = 1000
cluster_std = 20
subclass_std = 100
class_std = 2

data = []
labels_class = []
labels_cluster = []
for c, sizes in clusters_per_class.items():
    μ_c = np.random.randn(dims)*class_std
    for j, n in enumerate(sizes,1):
        μ_cs = μ_c + np.random.randn(dims)*subclass_std
        X = μ_cs + np.random.randn(n,dims)*cluster_std
        data.append(X)
        labels_class += [c]*n
        labels_cluster += [(c,j)]*n

X = np.vstack(data)
y_class = np.array(labels_class)
y_cluster = np.array(labels_cluster)

# ----------------------
# 2. Compute Scatter Matrices & Means
# ----------------------
def compute_hlda_stats(X, y_class, y_cluster, dims, reg=1e-8):
    N, d = X.shape
    μ_overall = X.mean(axis=0)
    classes = np.unique(y_class)
    S_B = np.zeros((d,d))
    S_W = np.zeros((d,d))
    parent_means = {}
    subclass_means = {}
    for c in classes:
        idx_c = np.where(y_class==c)[0]
        Xc = X[idx_c]
        μ_c = Xc.mean(axis=0)
        parent_means[c] = μ_c
        S_B += len(idx_c)*np.outer(μ_c-μ_overall, μ_c-μ_overall)
        # within-class
        S_W += (Xc-μ_c).T@(Xc-μ_c)
        # subclasses
        subs = np.unique([j for (cc,j) in y_cluster[idx_c]])
        subclass_means[c] = []
        for j in subs:
            idx_s = np.where((y_class==c)&(y_cluster[:,1]==j))[0]
            μ_cs = X[idx_s].mean(axis=0)
            subclass_means[c].append(μ_cs)
    S_W += reg*np.eye(d)
    return S_B, S_W, subclass_means, parent_means

# ----------------------
# 3. Distance‐Regularized hLDA: W‐only optimizer
# ----------------------
def optimize_W(S_B, S_W, subclass_means, parent_means,
               lambda1, lambda2,
               d, r=2,
               step_W=1e-4, num_iters=500, tol=1e-6, eps=1e-8):
    # initialize
    W = np.random.randn(d,r)
    W, _ = qr(W, mode='economic')
    for it in range(num_iters):
        # Rayleigh gradient
        f = np.trace(W.T@S_B@W)
        g = np.trace(W.T@S_W@W)+eps
        grad_W = (g*(2*S_B@W) - f*(2*S_W@W))/(g**2)
        # R1 gradient
        G1 = np.zeros_like(W)
        for c, subs in subclass_means.items():
            for i in range(len(subs)):
                for j in range(i+1,len(subs)):
                    d_ij = subs[i]-subs[j]
                    u = W.T@d_ij
                    norm_u = np.linalg.norm(u)+eps
                    G1 -= np.outer(d_ij, u)/(norm_u**2)
        # R2 gradient
        G2 = np.zeros_like(W)
        for c, subs in subclass_means.items():
            μ_c = parent_means[c]
            for μ_cs in subs:
                d_pc = μ_cs-μ_c
                v = W.T@d_pc
                norm_v = np.linalg.norm(v)+eps
                G2 += np.outer(d_pc, v)/norm_v
        # full gradient & step
        full_grad = grad_W + lambda1*G1 + lambda2*G2
        W_new = W + step_W*full_grad
        W, _ = qr(W_new, mode='economic')
        if np.linalg.norm(full_grad) < tol:
            break
    return W

# ----------------------
# 4. Train/Validation Split & Hyperparameter Grid
# ----------------------
X_train, X_val, y_train, y_val, yc_train, yc_val = train_test_split(
    X, y_class, y_cluster, test_size=0.2, stratify=y_class, random_state=0)

S_B_tr, S_W_tr, subs_tr, parents_tr = compute_hlda_stats(
    X_train, y_train, yc_train, dims)

lambda1_vals = [0.0, 1e-3,1e-2,1e-1,1.0]
lambda2_vals = [0.0, 1e-3,1e-2,1e-1,1.0]

best_acc = -np.inf
best_pair = (None,None)
best_W = None

for l1 in lambda1_vals:
    for l2 in lambda2_vals:
        W_cand = optimize_W(S_B_tr, S_W_tr, subs_tr, parents_tr,
                            l1, l2, dims)
        # project train -> centroids
        Y_tr = X_train@W_cand
        centroids = {c: Y_tr[y_train==c].mean(axis=0)
                     for c in np.unique(y_train)}
        # project val & classify
        Y_val = X_val@W_cand
        preds = np.array([
            min(centroids, key=lambda c: np.linalg.norm(y-centroids[c]))
            for y in Y_val
        ])
        acc = (preds==y_val).mean()
        if acc>best_acc:
            best_acc, best_pair, best_W = acc, (l1,l2), W_cand

print("Best λ₁, λ₂:", best_pair, "Validation acc:", best_acc)

# ----------------------
# 5. Re-fit on Full Data with Best Hyperparams
# ----------------------
S_B_full, S_W_full, subs_full, parents_full = compute_hlda_stats(
    X, y_class, y_cluster, dims)
W_final = optimize_W(S_B_full, S_W_full, subs_full, parents_full,
                     best_pair[0], best_pair[1], dims)

# ----------------------
# 6. Visualization
# ----------------------
Y = X@W_final
fig, axes = plt.subplots(1, len(np.unique(y_class)), figsize=(15,4))
for ax, c in zip(axes, np.unique(y_class)):
    idx = np.where(y_class==c)[0]
    sc = ax.scatter(Y[idx,0], Y[idx,1],
                    c=[jc for jc in y_cluster[idx,1]], cmap='tab10', alpha=0.7)
    ax.set_title(f"Class {c}")
plt.suptitle(f"Distance‐Regularized hLDA (λ₁={best_pair[0]}, λ₂={best_pair[1]})")
plt.tight_layout()
plt.show()
