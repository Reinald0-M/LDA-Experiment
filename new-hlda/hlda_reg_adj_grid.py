import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# ----------------------
# 1. Data Generation (Hierarchical Data)
# ----------------------
clusters_per_class = {1: [300], 2: [150, 150], 3: [75, 75, 75, 75]}
dims = 1000
class_std = 2
subclass_std = 100
cluster_std = 20
data_points = []    
labels_class = []    
labels_cluster = []  
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
# 2. Compute Scatter Matrices and Hierarchical Means
# ----------------------
N_total = data_points.shape[0]
overall_mean = np.mean(data_points, axis=0)
unique_classes = np.unique(labels_class)
d = dims
S_B = np.zeros((d, d))
for c in unique_classes:
    idx = np.where(labels_class == c)[0]
    X_c = data_points[idx, :]
    N_c = X_c.shape[0]
    mu_c = np.mean(X_c, axis=0)
    S_B += N_c * np.outer(mu_c - overall_mean, mu_c - overall_mean)
S_WS = np.zeros((d, d))
S_BS = np.zeros((d, d))
parent_means = {}
subclass_means = {}
for c in unique_classes:
    idx_class = np.where(labels_class == c)[0]
    X_c = data_points[idx_class, :]
    mu_c = np.mean(X_c, axis=0)
    parent_means[c] = mu_c
    class_subs = np.array([lbl[1] for lbl in labels_cluster[idx_class]])
    unique_subs = np.unique(class_subs)
    subclass_means[c] = []
    for sub in unique_subs:
        idx_sub = np.where(np.array([(lbl[0] == c and lbl[1] == sub) for lbl in labels_cluster]))[0]
        X_cs = data_points[idx_sub, :]
        N_cs = X_cs.shape[0]
        mu_cs = np.mean(X_cs, axis=0)
        subclass_means[c].append(mu_cs)
        diff = X_cs - mu_cs
        S_WS += diff.T @ diff
        S_BS += N_cs * np.outer(mu_cs - mu_c, mu_cs - mu_c)
reg = 1e-8

# ----------------------
# 3. Define the Hierarchical LDA Objective with Regularization Terms
# ----------------------
def hierarchical_lda_objective(W, S_B, S_W, subclass_means, parent_means, lambda1, lambda2, eps=1e-8):
    """
    Computes the hierarchical LDA objective:
      J*(W) = (tr(W^T S_B W))/(tr(W^T S_W W)) + lambda1 * R1(W) + lambda2 * R2(W)
    where:
      - R1(W) = sum_c sum_{i<j} 1/(||W^T(mu_{c,i}-mu_{c,j})|| + eps)
      - R2(W) = sum_c sum_i ||W^T (mu_{c,i}-mu_c)||
    """
    numerator = np.trace(W.T @ S_B @ W)
    denominator = np.trace(W.T @ S_W @ W)
    lda_obj = numerator / (denominator + eps)
    R1 = 0.0
    for c, sub_means in subclass_means.items():
        num_sub = len(sub_means)
        for i in range(num_sub):
            for j in range(i+1, num_sub):
                diff = W.T @ (sub_means[i] - sub_means[j])
                norm_val = np.linalg.norm(diff)
                R1 += 1.0 / (norm_val + eps)
    R2 = 0.0
    for c, sub_means in subclass_means.items():
        for mu_cs in sub_means:
            diff = W.T @ (mu_cs - parent_means[c])
            norm_val = np.linalg.norm(diff)
            R2 += norm_val
    return lda_obj + lambda1 * R1 + lambda2 * R2

# ----------------------
# 4. Optimize W by Maximizing the Hierarchical LDA Objective (Gradient Ascent)
# ----------------------
def optimize_W_hlda(S_B, S_W, subclass_means, parent_means, W_init, lambda1, lambda2, num_iters=200, step_size=1e-4, eps=1e-8):
    """
    Perform gradient ascent on the hierarchical LDA objective.
    Re-orthonormalize W (via QR) after each update.
    """
    W = W_init.copy()
    for it in range(num_iters):
        f = np.trace(W.T @ S_B @ W)
        g = np.trace(W.T @ S_W @ W) + eps
        grad_trace = (g * (2 * S_B @ W) - f * (2 * S_W @ W)) / (g**2)
        grad_R1 = np.zeros_like(W)
        for c, sub_means in subclass_means.items():
            num_sub = len(sub_means)
            for i in range(num_sub):
                for j in range(i+1, num_sub):
                    d_ij = sub_means[i] - sub_means[j]
                    proj = W.T @ d_ij
                    norm_proj = np.linalg.norm(proj)
                    grad_R1 += - (d_ij[:, None] @ (d_ij[:, None].T @ W)) / (norm_proj * (norm_proj + eps)**2 + eps)
        grad_R2 = np.zeros_like(W)
        for c, sub_means in subclass_means.items():
            for mu_cs in sub_means:
                d = mu_cs - parent_means[c]
                proj = W.T @ d
                norm_proj = np.linalg.norm(proj)
                grad_R2 += (d[:, None] @ (d[:, None].T @ W)) / (norm_proj + eps)
        grad = grad_trace + lambda1 * grad_R1 + lambda2 * grad_R2
        W += step_size * grad
        W, _ = np.linalg.qr(W)
        if it % 10 == 0:
            obj_val = hierarchical_lda_objective(W, S_B, S_W, subclass_means, parent_means, lambda1, lambda2)
            print(f"Iteration {it} (lambda1={lambda1}, lambda2={lambda2}): Objective = {obj_val:.4f}")
    return W

# ----------------------
# 5. Grid Search Over Independent lambda1, lambda2, and alpha, and Optimize W for Each
# ----------------------
r = 2
np.random.seed(2001)
W_init = np.random.randn(d, r)
grid_lambda1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
grid_lambda2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
grid_alpha = [0, 0.25, 0.5, 0.75, 1]
best_obj = -np.inf
best_l1 = None
best_l2 = None
best_alpha = None
best_W = None
for lam1 in grid_lambda1:
    for lam2 in grid_lambda2:
        for alpha in grid_alpha:
            S_W_current = alpha * S_WS + (1 - alpha) * S_BS
            S_W_current += reg * np.eye(d)
            W0 = W_init.copy()
            W_opt = optimize_W_hlda(S_B, S_W_current, subclass_means, parent_means, W0, lam1, lam2, num_iters=200, step_size=1e-4)
            obj_val = hierarchical_lda_objective(W_opt, S_B, S_W_current, subclass_means, parent_means, lam1, lam2)
            print(f"Grid search (lambda1={lam1}, lambda2={lam2}, alpha={alpha}): Final Objective = {obj_val:.4f}\n")
            if obj_val > best_obj:
                best_obj = obj_val
                best_l1 = lam1
                best_l2 = lam2
                best_alpha = alpha
                best_W = W_opt.copy()
print("Best hyperparameters:")
print("  lambda1 =", best_l1)
print("  lambda2 =", best_l2)
print("  alpha =", best_alpha)
print("Optimal hierarchical LDA objective value:", best_obj)

# ----------------------
# 6. Project the Data and Plot the Results Using the Best W
# ----------------------
S_W_best = best_alpha * S_WS + (1 - best_alpha) * S_BS
S_W_best += reg * np.eye(d)
data_points_2d = data_points @ best_W
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.tight_layout(pad=4.0)
num_classes = len(unique_classes)
discrete_cmap_all = plt.cm.get_cmap('viridis', num_classes)
ax = axs[0, 0]
scatter_all = ax.scatter(data_points_2d[:, 0], data_points_2d[:, 1], c=labels_class, cmap=discrete_cmap_all, alpha=0.7)
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_title("Hierarchical LDA Projection: All Classes")
cbar_all = fig.colorbar(scatter_all, ax=ax, ticks=unique_classes)
cbar_all.set_label('Class Label')
cbar_all.ax.set_yticklabels(unique_classes)
positions = [(0, 1), (1, 0), (1, 1)]
for pos, c in zip(positions, unique_classes):
    i, j = pos
    ax = axs[i, j]
    indices = np.where(labels_class == c)[0]
    data_class_2d = data_points_2d[indices, :]
    subclass_vals = np.array([lbl[1] for lbl in labels_cluster[indices]])
    unique_subclasses = np.unique(subclass_vals)
    discrete_cmap_sub = plt.cm.get_cmap('plasma', len(unique_subclasses))
    scatter_sub = ax.scatter(data_class_2d[:, 0], data_class_2d[:, 1], c=subclass_vals, cmap=discrete_cmap_sub, alpha=0.7)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"Class {c} (colored by subclass)")
    cbar_sub = fig.colorbar(scatter_sub, ax=ax, ticks=unique_subclasses)
    cbar_sub.set_label("Subclass (cluster index)")
    cbar_sub.ax.set_yticklabels(unique_subclasses)
plt.suptitle(f"Optimized Hierarchical LDA Projection\n(Best lambda1={best_l1}, lambda2={best_l2}, alpha={best_alpha})", fontsize=14)
plt.tight_layout()
plt.show()
