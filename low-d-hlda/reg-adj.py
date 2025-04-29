import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.model_selection import KFold

# ----------------------
# 1. Data Generation (Hierarchical Data)
# ----------------------
clusters_per_class = {1: [300], 2: [150, 150], 3: [75, 75, 75, 75]}
dims = 3
class_std = 1
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


plot = True
if plot:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        data_points[:, 0],
        data_points[:, 1],
        data_points[:, 2],
        c=labels_class,
        cmap='viridis',
        alpha=0.7
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Plot of Hierarchical Data by Parent Class")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1, ticks=np.unique(labels_class))
    cbar.set_label("Parent Class")
    plt.tight_layout()
    plt.show()


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
reg = 1e-8

# ----------------------
# 3. Hierarchical LDA Objective Function
# ----------------------
def hierarchical_lda_objective(W, S_B, S_W, subclass_means, parent_means, lambda1, lambda2, eps=1e-8):
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
# 4. Joint Gradient Ascent on W, lambda1, lambda2, and alpha
# ----------------------
def joint_gradient_ascent(S_B, S_WS, S_BS, subclass_means, parent_means, reg, num_iters=1000, step_W=1e-4, step_l1=1e-4, step_l2=1e-4, step_alpha=1e-4, eps=1e-8):
    W = np.random.randn(dims, 2)
    W, _ = np.linalg.qr(W)
    lambda1 = 0.5
    lambda2 = 0.5
    alpha = 0.5
    history = []
    gradW_history = []
    gradAlpha_history = []
    for it in range(num_iters):
        S_W = alpha * S_WS + (1 - alpha) * S_BS + reg * np.eye(dims)
        f = np.trace(W.T @ S_B @ W)
        g = np.trace(W.T @ S_W @ W) + eps
        grad_W = (g * (2 * S_B @ W) - f * (2 * S_W @ W)) / (g**2)
        grad_alpha = - (f / (g**2)) * np.trace(W.T @ (S_WS - S_BS) @ W)
        gradW_history.append(grad_W.copy())
        gradAlpha_history.append(grad_alpha)
        R1_val = 0.0
        for c, sub_means in subclass_means.items():
            num_sub = len(sub_means)
            for i in range(num_sub):
                for j in range(i+1, num_sub):
                    diff = W.T @ (sub_means[i] - sub_means[j])
                    R1_val += 1.0 / (np.linalg.norm(diff) + eps)
        R2_val = 0.0
        for c, sub_means in subclass_means.items():
            for mu_cs in sub_means:
                diff = W.T @ (mu_cs - parent_means[c])
                R2_val += np.linalg.norm(diff)
        grad_lambda1 = R1_val
        grad_lambda2 = R2_val
        W += step_W * grad_W
        W, _ = np.linalg.qr(W)
        lambda1 += step_l1 * grad_lambda1
        lambda2 += step_l2 * grad_lambda2
        alpha += step_alpha * grad_alpha
        lambda1 = max(lambda1, 0)
        lambda2 = max(lambda2, 0)
        alpha = np.clip(alpha, 0, 1)
        obj = hierarchical_lda_objective(W, S_B, S_W, subclass_means, parent_means, lambda1, lambda2)
        history.append(obj)
        if it % 50 == 0:
            print(f"Iter {it}: Obj={obj:.4f}, alpha={alpha:.4f}, lambda1={lambda1:.4f}, lambda2={lambda2:.4f}")
    return W, lambda1, lambda2, alpha, history, gradW_history, gradAlpha_history


W_opt, best_l1, best_l2, best_alpha, _ , gradW_history, gradAlpha_history = joint_gradient_ascent(S_B_full, S_WS_full, S_BS_full, subclass_means_full, parent_means_full, reg, num_iters=500)
print("Best Obj:", gradAlpha_history[-1])
print("Best alpha:", best_alpha)
print("Best lambda1:", best_l1)
print("Best lambda2:", best_l2)

def plot_full_projection(W, best_alpha, best_lambda1, best_lambda2, data_points, labels_class, labels_cluster, dims, reg, title_prefix="Full Data Projection"):
    # Compute the composite scatter matrix S_W using the best alpha.
    S_W = best_alpha * S_WS_full + (1 - best_alpha) * S_BS_full + reg * np.eye(dims)
    # Project the full data into the latent space.
    data_points_2d = data_points @ W
    unique_parents = np.unique(labels_class)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.tight_layout(pad=4.0)
    
    # Overall projection (all parent classes together)
    cmap_all = plt.cm.get_cmap('viridis', len(unique_parents))
    ax = axs[0, 0]
    sc_all = ax.scatter(data_points_2d[:, 0], data_points_2d[:, 1], c=labels_class, cmap=cmap_all, alpha=0.7)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Overall Projection")
    cbar = fig.colorbar(sc_all, ax=ax, ticks=unique_parents)
    cbar.set_label("Parent Class")
    
    # Projections for each parent class
    for (i, j), c in zip([(0, 1), (1, 0), (1, 1)], unique_parents):
        ax = axs[i, j]
        idx = np.where(labels_class == c)[0]
        X_c = data_points_2d[idx, :]
        subclass_vals = np.array([lbl[1] for lbl in labels_cluster[idx]])
        unique_subs = np.unique(subclass_vals)
        cmap_sub = plt.cm.get_cmap('plasma', len(unique_subs))
        sc = ax.scatter(X_c[:, 0], X_c[:, 1], c=subclass_vals, cmap=cmap_sub, alpha=0.7)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(f"Parent {c}")
        cbar_sub = fig.colorbar(sc, ax=ax, ticks=unique_subs)
        cbar_sub.set_label("Subclass")
    
    plt.suptitle(f"{title_prefix}\n(Best α = {best_alpha:.2f}, λ₁ = {best_lambda1:.2f}, λ₂ = {best_lambda2:.2f})", fontsize=16)
    plt.tight_layout()
    plt.show()

plot_full_projection(W_opt, best_alpha, best_l1, best_l2, data_points, labels_class, labels_cluster, dims, reg)



# ----------------------
# 5. k-Fold Cross Validation, Classification in Latent Space, and Plot
# ----------------------
cross_val = False
if cross_val:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for train_index, test_index in kf.split(data_points):
        X_train, X_test = data_points[train_index], data_points[test_index]
        y_train = labels_class[train_index]
        y_test = labels_class[test_index]
        cl_train = labels_cluster[train_index]
        cl_test = labels_cluster[test_index]
        S_B_train, S_WS_train, S_BS_train, parent_means_train, subclass_means_train = compute_scatter_matrices(X_train, y_train, cl_train, dims)
        W_fold, l1_fold, l2_fold, alpha_fold, _ = joint_gradient_ascent(S_B_train, S_WS_train, S_BS_train, subclass_means_train, parent_means_train, reg, num_iters=300)
        S_W_train = alpha_fold * S_WS_train + (1 - alpha_fold) * S_BS_train + reg * np.eye(dims)
        X_train_2d = X_train @ W_fold
        X_test_2d = X_test @ W_fold
        # Compute centroids in latent space for each subclass (within a parent)
        centroids = {}  # {parent: {subclass: centroid}}
        for c in np.unique(y_train):
            centroids[c] = {}
            idx = np.where(y_train == c)[0]
            X_c_2d = X_train_2d[idx, :]
            cl_c = cl_train[idx]
            unique_subs = np.unique([lbl[1] for lbl in cl_c])
            for sub in unique_subs:
                idx_sub = np.where(np.array([lbl[1] for lbl in cl_c]) == sub)[0]
                centroids[c][sub] = np.mean(X_c_2d[idx_sub, :], axis=0)
        # Classify test points using nearest centroid (within the same parent)
        pred_sub = []
        for i in range(len(X_test_2d)):
            p = y_test[i]  # parent class
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
        # Compute classification accuracy per parent class and overall for test samples
        accs = {}
        for c in np.unique(y_test):
            idx = np.where(y_test == c)[0]
            acc = np.mean(pred_sub[idx] == true_sub[idx])
            accs[c] = acc
        # Plot: For each parent class, we plot the training and test points in latent space colored by subclass,
        # with test points as stars; we annotate the plot with the classification accuracy.
        fig, axs = plt.subplots(1, len(np.unique(y_test)), figsize=(5*len(np.unique(y_test)), 5))
        if len(np.unique(y_test)) == 1:
            axs = [axs]
        for ax, c in zip(axs, np.unique(y_test)):
            idx_train = np.where(y_train == c)[0]
            idx_test = np.where(y_test == c)[0]
            X_c_train = X_train_2d[idx_train, :]
            X_c_test = X_test_2d[idx_test, :]
            subclass_train = np.array([lbl[1] for lbl in cl_train[idx_train]])
            subclass_test = np.array([lbl[1] for lbl in cl_test[idx_test]])
            discrete_cmap_sub = plt.cm.get_cmap('plasma', len(np.unique(subclass_train)))
            sc_tr = ax.scatter(X_c_train[:, 0], X_c_train[:, 1], c=subclass_train, cmap=discrete_cmap_sub, marker='o', alpha=0.7, label='Train')
            sc_te = ax.scatter(X_c_test[:, 0], X_c_test[:, 1], c=subclass_test, cmap=discrete_cmap_sub, marker='*', s=100, alpha=0.9, label='Test')
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title(f"Parent {c}: Accuracy={accs[c]*100:.1f}%")
            ax.legend()
        plt.suptitle(f"Fold {fold}: Projection per Parent Class (alpha={alpha_fold:.2f})", fontsize=16)
        plt.tight_layout()
        plt.show()
        fold += 1

# ----------------------
# 7. Plot Gradient History from Full Data Training
# ----------------------
fig_hist, axs_hist = plt.subplots(2, 1, figsize=(8, 6))
gradW_norms = [np.linalg.norm(g) for g in gradW_history]
axs_hist[0].plot(gradW_norms, label="Gradient Norm for W")
axs_hist[0].set_xlabel("Iteration")
axs_hist[0].set_ylabel("Norm")
axs_hist[0].set_title("Gradient Norm History for W")
axs_hist[0].legend()

axs_hist[1].plot(gradAlpha_history, label="Gradient for alpha", color='red')
axs_hist[1].set_xlabel("Iteration")
axs_hist[1].set_ylabel("Value")
axs_hist[1].set_title("Gradient History for alpha")
axs_hist[1].legend()

plt.tight_layout()
plt.show()
