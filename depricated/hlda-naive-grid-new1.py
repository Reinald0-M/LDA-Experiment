import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.linalg import eigh
import matplotlib.gridspec as gridspec

# 1. Generate Hierarchical Data
clusters_per_class = {1: [300], 2: [150, 150], 3: [75, 75, 75, 75]}
dims = 3
subclass_std = 100
cluster_std = 20
class_means = {
    1: np.pad([100], (0, dims - 1)),
    2: np.pad([0, 100], (0, dims - 2)),
    3: np.pad([0, 0, 100], (0, dims - 3))
}
data_points, labels_class, labels_cluster = [], [], []
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

# 2. Compute Scatter Matrices
overall_mean = np.mean(data_points, axis=0)
unique_classes = np.unique(labels_class)
S_B, S_WS, S_BS = np.zeros((dims, dims)), np.zeros((dims, dims)), np.zeros((dims, dims))
for c in unique_classes:
    idx = np.where(labels_class == c)[0]
    X_c = data_points[idx, :]
    mu_c = np.mean(X_c, axis=0)
    S_B += X_c.shape[0] * np.outer(mu_c - overall_mean, mu_c - overall_mean)
for c in unique_classes:
    idx_class = np.where(labels_class == c)[0]
    X_c = data_points[idx_class, :]
    mu_c = np.mean(X_c, axis=0)
    subclass_ids = np.unique([lbl[1] for lbl in labels_cluster[idx_class]])
    for sub in subclass_ids:
        idx_sub = np.where(np.all(labels_cluster == np.array((c, sub)), axis=1))[0]
        X_cs = data_points[idx_sub, :]
        mu_cs = np.mean(X_cs, axis=0)
        S_WS += (X_cs - mu_cs).T @ (X_cs - mu_cs)
        S_BS += X_cs.shape[0] * np.outer(mu_cs - mu_c, mu_cs - mu_c)

# 3. Grid Search Over Alpha, Beta
import matplotlib as mpl 

reg = 1e-6
grid_alpha = np.linspace(0, 1, 10)
grid_beta = np.linspace(0, 1, 10)
fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(len(grid_alpha), len(grid_beta), wspace=0.4, hspace=0.4)
num_classes = len(unique_classes)
cmap_all = ListedColormap(mpl.colormaps['viridis'](np.linspace(0, 1, num_classes)))


for i, alpha in enumerate(grid_alpha):
    for j, beta in enumerate(grid_alpha):
        beta = 1 - alpha
        S_num = alpha * S_B + beta * S_BS
        S_den = S_WS + reg * np.eye(dims)
        eigvals, eigvecs = eigh(S_num, S_den)
        top_indices = np.argsort(eigvals)[::-1][:2]
        W = eigvecs[:, top_indices]
        data_proj = data_points @ W

        ax = fig.add_subplot(gs[i, j])
        scatter = ax.scatter(data_proj[:, 0], data_proj[:, 1], c=labels_class, cmap=cmap_all, alpha=0.7)
        ax.set_title(f"α={alpha:.2f}, β={beta:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

for i, alpha in enumerate(grid_alpha):
    for j, beta in enumerate(grid_beta):
        S_num = alpha * S_B + beta * S_BS
        S_den = S_WS + reg * np.eye(dims)
        eigvals, eigvecs = eigh(S_num, S_den)
        top_indices = np.argsort(eigvals)[::-1][:2]
        W = eigvecs[:, top_indices]
        data_proj = data_points @ W

        ax = fig.add_subplot(gs[i, j])
        scatter = ax.scatter(data_proj[:, 0], data_proj[:, 1], c=labels_class, cmap=cmap_all, alpha=0.7)
        ax.set_title(f"α={alpha:.2f}, β={beta:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("Hierarchical LDA Projections for α, β Grid", fontsize=20)
plt.show()
