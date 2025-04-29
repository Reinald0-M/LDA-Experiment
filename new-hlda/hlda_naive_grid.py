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
class_means = {
    1: np.pad([100], (0, dims - 1)),
    2: np.pad([0, 100], (0, dims - 2)),
    3: np.pad([0, 0, 100], (0, dims - 3))
}
data_points = []    
labels_class = []    
labels_cluster = []
for class_label, cluster_sizes in clusters_per_class.items():
    base_mean = class_means[class_label]
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
N_total = data_points.shape[0]
overall_mean = np.mean(data_points, axis=0)
unique_classes = np.unique(labels_class)
dims = data_points.shape[1]
S_B = np.zeros((dims, dims))
S_WS = np.zeros((dims, dims))
S_BS = np.zeros((dims, dims))
for c in unique_classes:
    idx = np.where(labels_class == c)[0]
    X_c = data_points[idx, :]
    N_c = X_c.shape[0]
    mu_c = np.mean(X_c, axis=0)
    S_B += N_c * np.outer(mu_c - overall_mean, mu_c - overall_mean)
for c in unique_classes:
    idx_class = np.where(labels_class == c)[0]
    X_c = data_points[idx_class, :]
    mu_c = np.mean(X_c, axis=0)
    subclass_ids = np.unique([lbl[1] for lbl in labels_cluster[idx_class]])
    for sub in subclass_ids:
        idx_sub = np.where(np.all(labels_cluster == np.array((c, sub)), axis=1))[0]
        X_cs = data_points[idx_sub, :]
        N_cs = X_cs.shape[0]
        mu_cs = np.mean(X_cs, axis=0)
        diff = X_cs - mu_cs
        S_WS += diff.T @ diff
        S_BS += N_cs * np.outer(mu_cs - mu_c, mu_cs - mu_c)

# ----------------------
# 3. Grid Search Over Alpha
# ----------------------
reg = 1e-6
best_obj = -np.inf
best_alpha = None
best_W = None
grid_alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for alpha in grid_alpha:
    S_W_mod = alpha * S_WS + (1 - alpha) * S_BS
    S_W_mod += reg * np.eye(dims)
    eigvals, eigvecs = eigh(S_B, S_W_mod)
    top_indices = np.argsort(eigvals)[::-1][:2]
    W = eigvecs[:, top_indices]
    obj_val = np.trace(W.T @ S_B @ W) / np.trace(W.T @ S_W_mod @ W)
    if obj_val > best_obj:
        best_obj = obj_val
        best_alpha = alpha
        best_W = W.copy()

# ----------------------
# 4. Plot the Best Projection
# ----------------------
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
    idx = np.where(labels_class == c)[0]
    X_c_2d = data_points_2d[idx, :]
    subclass_vals = np.array([lbl[1] for lbl in labels_cluster[idx]])
    unique_subs = np.unique(subclass_vals)
    discrete_cmap_sub = plt.cm.get_cmap('plasma', len(unique_subs))
    scatter_sub = ax.scatter(X_c_2d[:, 0], X_c_2d[:, 1], c=subclass_vals, cmap=discrete_cmap_sub, alpha=0.7)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"Class {c} (colored by subclass)")
    cbar_sub = fig.colorbar(scatter_sub, ax=ax, ticks=unique_subs)
    cbar_sub.set_label("Subclass (cluster index)")
    cbar_sub.ax.set_yticklabels(unique_subs)
plt.suptitle(f"Hierarchical LDA Projection\n(Best $\\alpha$ = {best_alpha:.2f})", fontsize=24)
plt.tight_layout()
plt.show()
