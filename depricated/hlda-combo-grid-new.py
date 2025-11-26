import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
# ----------------------
# 1. Data Generation
# ----------------------
clusters_per_class = {1: [300], 2: [150, 150], 3: [75, 75, 75, 75]}
dims = 3
class_std, subclass_std, cluster_std = 2, 100, 20
data_points, labels_class, labels_cluster = [], [], []
class_means = {}

for class_label, cluster_sizes in clusters_per_class.items():
    base_mean = np.random.randn(dims) * class_std
    class_means[class_label] = base_mean
    for cluster_index, n_points in enumerate(cluster_sizes, start=1):
        subclass_mean = base_mean + np.random.randn(dims) * subclass_std
        points = subclass_mean + np.random.randn(n_points, dims) * cluster_std
        data_points.append(points)
        labels_class.extend([class_label] * n_points)
        labels_cluster.extend([(class_label, cluster_index)] * n_points)

data_points = np.vstack(data_points)
labels_class = np.array(labels_class)
labels_cluster = np.array(labels_cluster)

# ----------------------
# 2. Compute Scatter Matrices
# ----------------------
overall_mean = np.mean(data_points, axis=0)
unique_classes = np.unique(labels_class)
d = dims
S_B, S_WS, S_BS = np.zeros((d, d)), np.zeros((d, d)), np.zeros((d, d))
parent_means, subclass_means = {}, {}

for c in unique_classes:
    idx = np.where(labels_class == c)[0]
    X_c = data_points[idx]
    mu_c = np.mean(X_c, axis=0)
    parent_means[c] = mu_c
    N_c = len(idx)
    S_B += N_c * np.outer(mu_c - overall_mean, mu_c - overall_mean)
    sub_ids = np.array([lbl[1] for lbl in labels_cluster[idx]])
    subclass_means[c] = []
    for sub in np.unique(sub_ids):
        idx_sub = np.where(np.all(labels_cluster == np.array((c, sub)), axis=1))[0]
        X_cs = data_points[idx_sub]
        mu_cs = np.mean(X_cs, axis=0)
        subclass_means[c].append(mu_cs)
        S_WS += (X_cs - mu_cs).T @ (X_cs - mu_cs)
        S_BS += len(idx_sub) * np.outer(mu_cs - mu_c, mu_cs - mu_c)

# ----------------------
# 3. Objective
# ----------------------
def hierarchical_lda_objective(W, S_B, S_BS, S_WS, subclass_means, parent_means, alpha, beta, lambda1, lambda2, eps=1e-8):
    S_num = alpha * S_B + beta * S_BS
    num = np.trace(W.T @ S_num @ W)
    den = np.trace(W.T @ S_WS @ W) + eps
    lda_term = num / den
    R1, R2 = 0.0, 0.0
    for c, subs in subclass_means.items():
        for i in range(len(subs)):
            for j in range(i+1, len(subs)):
                diff = W.T @ (subs[i] - subs[j])
                R1 += 1. / (np.linalg.norm(diff) + eps)
        for mu_cs in subs:
            R2 += np.linalg.norm(W.T @ (mu_cs - parent_means[c]))
    return lda_term - lambda1 * R1 - lambda2 * R2

# ----------------------
# 4. Optimization
# ----------------------
def optimize_W(S_B, S_BS, S_WS, subclass_means, parent_means, W_init, alpha, beta, lambda1, lambda2, steps=200, lr=1e-4, eps=1e-8):
    W = W_init.copy()
    S_num = alpha * S_B + beta * S_BS
    for _ in range(steps):
        f = np.trace(W.T @ S_num @ W)
        g = np.trace(W.T @ S_WS @ W) + eps
        grad_trace = (g * (2 * S_num @ W) - f * (2 * S_WS @ W)) / (g ** 2)
        grad_R1, grad_R2 = np.zeros_like(W), np.zeros_like(W)
        for c, subs in subclass_means.items():
            for i in range(len(subs)):
                for j in range(i+1, len(subs)):
                    d = subs[i] - subs[j]
                    proj = W.T @ d
                    norm_proj = np.linalg.norm(proj)
                    grad_R1 += - (d[:, None] @ (d[:, None].T @ W)) / (norm_proj * (norm_proj + eps)**2 + eps)
            for mu_cs in subs:
                d = mu_cs - parent_means[c]
                proj = W.T @ d
                norm_proj = np.linalg.norm(proj)
                grad_R2 += (d[:, None] @ (d[:, None].T @ W)) / (norm_proj + eps)
        grad = grad_trace + lambda1 * grad_R1 + lambda2 * grad_R2
        W += lr * grad
        W, _ = np.linalg.qr(W)
    return W

# ----------------------
# 5. Grid Search + Plot
# ----------------------
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import itertools
# [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
np.random.seed(2001)
r = 2
W_init = np.random.randn(d, r)
W_init, _ = np.linalg.qr(W_init)
grid_alpha = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
grid_beta = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
grid_lambda1 =[ 0.7, 0.9, 1]
grid_lambda2 =[ 0.7, 0.9, 1]
best_results = []

fig, axs = plt.subplots(len(grid_alpha), len(grid_beta), figsize=(16, 12))
fig.suptitle("Hierarchical LDA Projections for (α, β)", fontsize=18)
plot_idx = list(itertools.product(range(len(grid_alpha)), range(len(grid_beta))))

for i, alpha in enumerate(grid_alpha):
    for j, beta in enumerate(grid_beta):
        best_score = -np.inf
        best_W = None
        best_config = None
        for l1 in grid_lambda1:
            for l2 in grid_lambda2:
                print(f'Optimizing for α={alpha}, β={beta}, λ1={l1}, λ2={l2}')
                SWS_reg = S_WS + 1e-6 * np.eye(d)
                W_opt = optimize_W(S_B, S_BS, SWS_reg, subclass_means, parent_means,
                                   W_init.copy(), alpha, beta, l1, l2)
                score = hierarchical_lda_objective(W_opt, S_B, S_BS, SWS_reg,
                                                   subclass_means, parent_means,
                                                   alpha, beta, l1, l2)
                if score > best_score:
                    best_score = score
                    best_W = W_opt
                    best_config = (alpha, beta, l1, l2)

        proj = data_points @ best_W
        ax = axs[i, j]
        cmap = ListedColormap(cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_classes))))
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels_class, cmap=cmap, alpha=0.6)
        ax.set_title(f"α={alpha}, β={beta}\nλ1={best_config[2]}, λ2={best_config[3]}")
        ax.set_xlabel("Comp 1")
        ax.set_ylabel("Comp 2")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
