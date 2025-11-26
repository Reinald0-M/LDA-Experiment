import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.model_selection import KFold

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
def compute_scatter_matrices(data, labels_class, labels_cluster, dims):
    overall_mean = np.mean(data, axis=0)
    unique_classes = np.unique(labels_class)
    S_B = np.zeros((dims, dims))
    S_WS = np.zeros((dims, dims))
    S_BS = np.zeros((dims, dims))
    parent_means = {}
    subclass_means = {}
    for c in unique_classes:
        idx = np.where(labels_class == c)[0]
        X_c = data[idx, :]
        N_c = X_c.shape[0]
        mu_c = np.mean(X_c, axis=0)
        parent_means[c] = mu_c
        S_B += N_c * np.outer(mu_c - overall_mean, mu_c - overall_mean)
        class_subs = np.array([lbl[1] for lbl in labels_cluster[idx]])
        unique_subs = np.unique(class_subs)
        subclass_means[c] = []
        for sub in unique_subs:
            idx_sub = np.where(np.array([(lbl[0] == c and lbl[1] == sub) for lbl in labels_cluster]))[0]
            X_cs = data[idx_sub, :]
            N_cs = X_cs.shape[0]
            mu_cs = np.mean(X_cs, axis=0)
            subclass_means[c].append(mu_cs)
            diff = X_cs - mu_cs
            S_WS += diff.T @ diff
            S_BS += N_cs * np.outer(mu_cs - mu_c, mu_cs - mu_c)
    return S_B, S_WS, S_BS, parent_means, subclass_means

S_B_full, S_WS_full, S_BS_full, parent_means_full, subclass_means_full = compute_scatter_matrices(data_points, labels_class, labels_cluster, dims)
reg = 1e-6

# ----------------------
# 3. Hierarchical LDA Objective Function
# ----------------------
def hierarchical_lda_objective(W, S_B, S_W, subclass_means, parent_means, lambda_val, eps=1e-8):
    numerator = np.trace(W.T @ S_B @ W)
    denominator = np.trace(W.T @ S_W @ W)
    return lambda_val * (numerator / (denominator + eps)) + (1 - lambda_val) * (numerator / (denominator + eps))

def objective(W, alpha, S_B, S_WS, S_BS, reg, eps=1e-8):
    S_W = alpha * S_WS + (1 - alpha) * S_BS + reg * np.eye(dims)
    f = np.trace(W.T @ S_B @ W)
    g = np.trace(W.T @ S_W @ W)
    return f / (g + eps), S_W, f, g

# ----------------------
# 4. Joint Gradient Ascent on W and α (with Gradient History)
# ----------------------
def joint_gradient_ascent(S_B, S_WS, S_BS, subclass_means, parent_means, reg, num_iters=5000, step_W=1e-4, step_alpha=1e-4, eps=1e-8):
    W = np.random.randn(dims, 2)
    W, _ = np.linalg.qr(W)
    alpha = 0.5
    obj_history = []
    gradW_history = []
    gradAlpha_history = []
    for it in range(num_iters):
        J, S_W, f, g = objective(W, alpha, S_B, S_WS, S_BS, reg)
        grad_W = (g * (2 * S_B @ W) - f * (2 * S_W @ W)) / (g**2)
        A = np.trace(W.T @ S_WS @ W)
        B = np.trace(W.T @ S_BS @ W)
        grad_alpha = - (f / (g**2)) * (A - B)
        W += step_W * grad_W
        W, _ = np.linalg.qr(W)
        alpha += step_alpha * grad_alpha
        alpha = np.clip(alpha, 0, 1)
        obj_history.append(J)
        gradW_history.append(np.linalg.norm(grad_W))
        gradAlpha_history.append(np.linalg.norm(grad_alpha))
        if it % 50 == 0:
            print(f"Iteration {it}: Obj={J:.4f}, α={alpha:.4f}")
    return W, alpha, obj_history, gradW_history, gradAlpha_history

W_full, best_alpha, history, gradW_history, gradAlpha_history = joint_gradient_ascent(S_B_full, S_WS_full, S_BS_full, subclass_means_full, parent_means_full, reg, num_iters=5000)
print("Best objective (full data):", history[-1])
print("Best α (full data):", best_alpha)

# ----------------------
# 5. Plot Full Data Projection in the Optimal Latent Space (2x2 Plot)
# ----------------------
S_W_best = best_alpha * S_WS_full + (1 - best_alpha) * S_BS_full + reg * np.eye(dims)
data_points_2d = data_points @ W_full
fig_full, axs_full = plt.subplots(2, 2, figsize=(12, 10))
fig_full.tight_layout(pad=4.0)
unique_parents = np.unique(labels_class)
cmap_all = plt.cm.get_cmap('viridis', len(unique_parents))
ax0 = axs_full[0, 0]
sc_all = ax0.scatter(data_points_2d[:, 0], data_points_2d[:, 1], c=labels_class, cmap=cmap_all, alpha=0.7)
ax0.set_title("Overall Projection")
ax0.set_xlabel("Component 1")
ax0.set_ylabel("Component 2")
cbar_all = fig_full.colorbar(sc_all, ax=ax0, ticks=unique_parents)
cbar_all.set_label("Parent Class")
for (i,j), c in zip([(0,1), (1,0), (1,1)], unique_parents):
    ax = axs_full[i,j]
    idx = np.where(labels_class == c)[0]
    X_c = data_points_2d[idx, :]
    subclass_vals = np.array([lbl[1] for lbl in labels_cluster[idx]])
    unique_subs = np.unique(subclass_vals)
    cmap_sub = plt.cm.get_cmap('plasma', len(unique_subs))
    sc_c = ax.scatter(X_c[:, 0], X_c[:, 1], c=subclass_vals, cmap=cmap_sub, alpha=0.7)
    ax.set_title(f"Parent {c}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    cbar_c = fig_full.colorbar(sc_c, ax=ax, ticks=unique_subs)
    cbar_c.set_label("Subclass")
plt.suptitle(f"Full Data Projection (Best α = {best_alpha:.2f})", fontsize=16)
plt.tight_layout()
plt.show()

# ----------------------
# 6. k-Fold Cross Validation, Nearest-Centroid Classification, and Plot
# ----------------------
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold = 1
for train_index, test_index in kf.split(data_points):
    X_train, X_test = data_points[train_index], data_points[test_index]
    y_train, y_test = labels_class[train_index], labels_class[test_index]
    cl_train, cl_test = labels_cluster[train_index], labels_cluster[test_index]
    S_B_train, S_WS_train, S_BS_train, parent_means_train, subclass_means_train = compute_scatter_matrices(X_train, y_train, cl_train, dims)
    W_fold, alpha_fold, _ , _ , _ = joint_gradient_ascent(S_B_train, S_WS_train, S_BS_train, subclass_means_train, parent_means_train, reg, num_iters=300)
    S_W_train = alpha_fold * S_WS_train + (1 - alpha_fold) * S_BS_train + reg * np.eye(dims)
    X_train_2d = X_train @ W_fold
    X_test_2d = X_test @ W_fold
    centroids = {}
    for c in np.unique(y_train):
        centroids[c] = {}
        idx = np.where(y_train == c)[0]
        X_c_train = X_train_2d[idx, :]
        cl_c_train = cl_train[idx]
        unique_subs = np.unique([lbl[1] for lbl in cl_c_train])
        for sub in unique_subs:
            idx_sub = np.where(np.array([lbl[1] for lbl in cl_c_train]) == sub)[0]
            centroids[c][sub] = np.mean(X_c_train[idx_sub, :], axis=0)
    pred_sub = []
    for i in range(len(X_test_2d)):
        p = y_test[i]
        x = X_test_2d[i, :]
        best_sub = None
        best_dist = np.inf
        for sub, center in centroids[p].items():
            dist = np.linalg.norm(x - center)
            if dist < best_dist:
                best_dist = dist
                best_sub = sub
        pred_sub.append(best_sub)
    pred_sub = np.array(pred_sub)
    true_sub = np.array([lbl[1] for lbl in cl_test])
    accs = {}
    for c in np.unique(y_test):
        idx = np.where(y_test == c)[0]
        accs[c] = np.mean(pred_sub[idx] == true_sub[idx])
    parents = np.unique(y_train)
    fig_cv, axs_cv = plt.subplots(1, len(parents), figsize=(5*len(parents), 5))
    if len(parents) == 1:
        axs_cv = [axs_cv]
    for ax, c in zip(axs_cv, parents):
        idx_tr = np.where(y_train == c)[0]
        idx_te = np.where(y_test == c)[0]
        X_tr = X_train_2d[idx_tr, :]
        X_te = X_test_2d[idx_te, :]
        subclass_tr = np.array([lbl[1] for lbl in cl_train[idx_tr]])
        subclass_te = np.array([lbl[1] for lbl in cl_test[idx_te]])
        cmap_sub = plt.cm.get_cmap('plasma', len(np.unique(subclass_tr)))
        sc_tr = ax.scatter(X_tr[:, 0], X_tr[:, 1], c=subclass_tr, cmap=cmap_sub, marker='o', alpha=0.7, label='Train')
        sc_te = ax.scatter(X_te[:, 0], X_te[:, 1], c=subclass_te, cmap=cmap_sub, marker='*', s=100, alpha=0.9, label='Test')
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(f"Parent {c}: Acc={accs[c]*100:.1f}%")
        ax.legend()
    plt.suptitle(f"Fold {fold}: Projection per Parent (α={alpha_fold:.2f})", fontsize=16)
    plt.tight_layout()
    plt.show()
    fold += 1

# ----------------------
# 7. Plot Gradient History from Full Data Training
# ----------------------
fig_hist, axs_hist = plt.subplots(2, 1, figsize=(8, 6))
axs_hist[0].plot(gradW_history, label="Gradient Norm for W")
axs_hist[0].set_xlabel("Iteration")
axs_hist[0].set_ylabel("Norm")
axs_hist[0].set_title("Gradient Norm History for W")
axs_hist[0].legend()
axs_hist[1].plot(gradAlpha_history, label="Gradient Norm for α", color='red')
axs_hist[1].set_xlabel("Iteration")
axs_hist[1].set_ylabel("Norm")
axs_hist[1].set_title("Gradient Norm History for α")
axs_hist[1].legend()
plt.tight_layout()
plt.show()
