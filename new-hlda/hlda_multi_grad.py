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
# 2. Compute Scatter Matrices
# ----------------------
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
S_W_reg = np.zeros((d, d))
parent_means = {}
for c in unique_classes:
    idx = np.where(labels_class == c)[0]
    X_c = data_points[idx, :]
    mu_c = np.mean(X_c, axis=0)
    parent_means[c] = mu_c
    diff = X_c - mu_c
    S_W_reg += diff.T @ diff
reg = 1e-8
S_W_reg += reg * np.eye(d)
local_S_B = {}
local_S_W = {}
subclass_means = {}
for c in unique_classes:
    idx = np.where(labels_class == c)[0]
    X_c = data_points[idx, :]
    mu_c = np.mean(X_c, axis=0)
    subclass_means[c] = []
    class_subs = np.array([lbl[1] for lbl in labels_cluster[idx]])
    unique_subs = np.unique(class_subs)
    S_B_c = np.zeros((d, d))
    S_W_c = np.zeros((d, d))
    for sub in unique_subs:
        idx_sub = np.where(np.array([(lbl[0] == c and lbl[1] == sub) for lbl in labels_cluster]))[0]
        X_cs = data_points[idx_sub, :]
        N_cs = X_cs.shape[0]
        mu_cs = np.mean(X_cs, axis=0)
        subclass_means[c].append(mu_cs)
        diff = X_cs - mu_cs
        S_W_c += diff.T @ diff
        S_B_c += N_cs * np.outer(mu_cs - mu_c, mu_cs - mu_c)
    local_S_B[c] = S_B_c
    local_S_W[c] = S_W_c

# ----------------------
# 3. Multi-Level Scatter Objective Function
# ----------------------
def multi_level_scatter_objective(W, lambda_val, S_B, S_W_reg, local_S_B, local_S_W, eps=1e-8):
    global_term = np.trace(W.T @ S_B @ W) / (np.trace(W.T @ S_W_reg @ W) + eps)
    local_term_sum = 0.0
    for c in local_S_B.keys():
        term = np.trace(W.T @ local_S_B[c] @ W) / (np.trace(W.T @ local_S_W[c] @ W) + eps)
        local_term_sum += term
    return lambda_val * global_term + (1 - lambda_val) * local_term_sum

# ----------------------
# 4. Optimize W via Gradient Ascent with Gradient History (for lambda1, lambda2 version)
# ----------------------
def optimize_W_hlda(S_B, S_W_reg, local_S_B, local_S_W, W_init, lambda_val, num_iters=2000, step_size=1e-4, eps=1e-8):
    W = W_init.copy()
    obj_history = []
    grad_norm_history = []
    for i in range(num_iters):
        f = np.trace(W.T @ S_B @ W)
        g = np.trace(W.T @ S_W_reg @ W) + eps
        grad_global = (g * (2 * S_B @ W) - f * (2 * S_W_reg @ W)) / (g**2)
        grad_local = np.zeros_like(W)
        for c in local_S_B.keys():
            f_c = np.trace(W.T @ local_S_B[c] @ W)
            g_c = np.trace(W.T @ local_S_W[c] @ W) + eps
            grad_local += (g_c * (2 * local_S_B[c] @ W) - f_c * (2 * local_S_W[c] @ W)) / (g_c**2)
        grad = lambda_val * grad_global + (1 - lambda_val) * grad_local
        W += step_size * grad
        W, _ = np.linalg.qr(W)
        obj = multi_level_scatter_objective(W, lambda_val, S_B, S_W_reg, local_S_B, local_S_W)
        obj_history.append(obj)
        grad_norm_history.append(np.linalg.norm(grad))
        if i % 50 == 0:
            print(f"Iteration {i}: Obj = {obj:.4f}")
    return W, obj_history, grad_norm_history

r = 2
np.random.seed(2001)
W_init = np.random.randn(d, r)
lambda_val = 0.5
if np.abs(lambda_val - 1) < 1e-6:
    reg_mat = 1e-6 * np.eye(d)
    eigvals, eigvecs = eigh(S_B, S_W_reg + reg_mat)
    W_opt = eigvecs[:, -r:]
    print("Using closed-form solution (λ=1).")
else:
    W_opt, obj_history, grad_norm_history = optimize_W_hlda(S_B, S_W_reg, local_S_B, local_S_W, W_init, lambda_val, num_iters=5000, step_size=1e-4)
    print("Optimization finished.")
obj_opt = multi_level_scatter_objective(W_opt, lambda_val, S_B, S_W_reg, local_S_B, local_S_W)
print("Optimal objective value:", obj_opt)

# Plot gradient norm history
plt.figure(figsize=(8, 6))
plt.plot(grad_norm_history, label='Grad Norm')
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm History")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------
# 5. Project the Data and Plot the Results Using the Best W
# ----------------------
data_points_2d = data_points @ W_opt
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.tight_layout(pad=4.0)
num_classes = len(unique_classes)
discrete_cmap_all = plt.cm.get_cmap('viridis', num_classes)
ax = axs[0, 0]
scatter_all = ax.scatter(data_points_2d[:, 0], data_points_2d[:, 1], c=labels_class, cmap=discrete_cmap_all, alpha=0.7)
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_title("Multi-Level LDA Projection: All Classes")
cbar_all = fig.colorbar(scatter_all, ax=ax, ticks=unique_classes)
cbar_all.set_label('Class Label')
cbar_all.ax.set_yticklabels(unique_classes)
positions = [(0, 1), (1, 0), (1, 1)]
for pos, c in zip(positions, unique_classes):
    i, j = pos
    ax = axs[i, j]
    idx = np.where(labels_class == c)[0]
    data_c_2d = data_points_2d[idx]
    subclass_vals = np.array([lbl[1] for lbl in labels_cluster[idx]])
    unique_subclasses = np.unique(subclass_vals)
    discrete_cmap_sub = plt.cm.get_cmap('plasma', len(unique_subclasses))
    scatter_sub = ax.scatter(data_c_2d[:, 0], data_c_2d[:, 1], c=subclass_vals, cmap=discrete_cmap_sub, alpha=0.7)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"Class {c} (colored by subclass)")
    cbar_sub = fig.colorbar(scatter_sub, ax=ax, ticks=unique_subclasses)
    cbar_sub.set_label("Subclass (cluster index)")
    cbar_sub.ax.set_yticklabels(unique_subclasses)
plt.suptitle(f"Optimized Multi-Level LDA Projection (λ = {lambda_val})", fontsize=14)
plt.tight_layout()
plt.show()
