# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.linalg import eigh

# # ----------------------
# # 1. Data Generation (Hierarchical Data)
# # ----------------------
# clusters_per_class = {1: [300], 2: [150, 150], 3: [75, 75, 75, 75]}
# dims = 1000
# class_std = 2
# subclass_std = 100
# cluster_std = 20
# data_points = []    
# labels_class = []    
# labels_cluster = []  
# class_means = {}
# for class_label, cluster_sizes in clusters_per_class.items():
#     base_mean = np.random.randn(dims) * class_std
#     class_means[class_label] = base_mean
#     for cluster_index, n_points in enumerate(cluster_sizes, start=1):
#         subclass_mean = base_mean + np.random.randn(dims) * subclass_std
#         points = subclass_mean + np.random.randn(n_points, dims) * cluster_std
#         data_points.append(points)
#         labels_class.extend([class_label] * n_points)
#         labels_cluster.extend([(class_label, cluster_index)] * n_points)
# data_points = np.vstack(data_points)
# labels_class = np.array(labels_class)
# labels_cluster = np.array(labels_cluster)

# # ----------------------
# # 2. Compute Scatter Matrices and Hierarchical Means
# # ----------------------
# N_total = data_points.shape[0]
# overall_mean = np.mean(data_points, axis=0)
# unique_classes = np.unique(labels_class)
# d = dims
# S_B = np.zeros((d, d))
# for c in unique_classes:
#     idx = np.where(labels_class == c)[0]
#     X_c = data_points[idx, :]
#     N_c = X_c.shape[0]
#     mu_c = np.mean(X_c, axis=0)
#     S_B += N_c * np.outer(mu_c - overall_mean, mu_c - overall_mean)
# S_W = np.zeros((d, d))
# for c in unique_classes:
#     idx = np.where(labels_class == c)[0]
#     X_c = data_points[idx, :]
#     mu_c = np.mean(X_c, axis=0)
#     diff = X_c - mu_c
#     S_W += diff.T @ diff
# reg = 1e-8
# S_W += reg * np.eye(d)
# parent_means = {}
# subclass_means = {}
# for c in unique_classes:
#     idx_class = np.where(labels_class == c)[0]
#     X_c = data_points[idx_class, :]
#     mu_c = np.mean(X_c, axis=0)
#     parent_means[c] = mu_c
#     class_subs = np.array([lbl[1] for lbl in labels_cluster[idx_class]])
#     unique_subs = np.unique(class_subs)
#     subclass_means[c] = []
#     for sub in unique_subs:
#         idx_sub = np.where(np.array([(lbl[0] == c and lbl[1] == sub) for lbl in labels_cluster]))[0]
#         X_cs = data_points[idx_sub, :]
#         mu_cs = np.mean(X_cs, axis=0)
#         subclass_means[c].append(mu_cs)

# # ----------------------
# # 3. Hierarchical LDA Objective Function
# # ----------------------
# def hierarchical_lda_objective(W, S_B, S_W, subclass_means, parent_means, lambda1, lambda2, eps=1e-8):
#     numerator = np.trace(W.T @ S_B @ W)
#     denominator = np.trace(W.T @ S_W @ W)
#     lda_obj = numerator / (denominator + eps)
#     R1 = 0.0
#     for c, sub_means in subclass_means.items():
#         num_sub = len(sub_means)
#         for i in range(num_sub):
#             for j in range(i+1, num_sub):
#                 diff = W.T @ (sub_means[i] - sub_means[j])
#                 norm_val = np.linalg.norm(diff)
#                 R1 += 1.0 / (norm_val + eps)
#     R2 = 0.0
#     for c, sub_means in subclass_means.items():
#         for mu_cs in sub_means:
#             diff = W.T @ (mu_cs - parent_means[c])
#             norm_val = np.linalg.norm(diff)
#             R2 += norm_val
#     return lda_obj + lambda1 * R1 + lambda2 * R2

# # ----------------------
# # 4. Joint Gradient Ascent on W, lambda1, and lambda2
# # ----------------------
# def joint_gradient_ascent(S_B, S_W, subclass_means, parent_means, reg, num_iters=500, step_W=1e-4, step_l1=1e-4, step_l2=1e-4, eps=1e-8):
#     W = np.random.randn(d, 2)
#     W, _ = np.linalg.qr(W)
#     lambda1 = 0.5
#     lambda2 = 0.5
#     for it in range(num_iters):
#         f = np.trace(W.T @ S_B @ W)
#         g = np.trace(W.T @ S_W @ W) + eps
#         grad_trace = (g * (2 * S_B @ W) - f * (2 * S_W @ W)) / (g**2)
#         grad_R1 = np.zeros_like(W)
#         for c, sub_means in subclass_means.items():
#             num_sub = len(sub_means)
#             for i in range(num_sub):
#                 for j in range(i+1, num_sub):
#                     d_ij = sub_means[i] - sub_means[j]
#                     proj = W.T @ d_ij
#                     norm_proj = np.linalg.norm(proj)
#                     grad_R1 += - (d_ij[:, None] @ (d_ij[:, None].T @ W)) / (norm_proj * (norm_proj + eps)**2 + eps)
#         grad_R2 = np.zeros_like(W)
#         for c, sub_means in subclass_means.items():
#             for mu_cs in sub_means:
#                 d_vec = mu_cs - parent_means[c]
#                 proj = W.T @ d_vec
#                 norm_proj = np.linalg.norm(proj)
#                 grad_R2 += (d_vec[:, None] @ (d_vec[:, None].T @ W)) / (norm_proj + eps)
#         grad_W = grad_trace + lambda1 * grad_R1 + lambda2 * grad_R2
#         W += step_W * grad_W
#         W, _ = np.linalg.qr(W)
#         R1_val = 0.0
#         for c, sub_means in subclass_means.items():
#             num_sub = len(sub_means)
#             for i in range(num_sub):
#                 for j in range(i+1, num_sub):
#                     diff = W.T @ (sub_means[i] - sub_means[j])
#                     R1_val += 1.0 / (np.linalg.norm(diff) + eps)
#         R2_val = 0.0
#         for c, sub_means in subclass_means.items():
#             for mu_cs in sub_means:
#                 diff = W.T @ (mu_cs - parent_means[c])
#                 R2_val += np.linalg.norm(diff)
#         grad_lambda1 = R1_val
#         grad_lambda2 = R2_val
#         lambda1 += step_l1 * grad_lambda1
#         lambda2 += step_l2 * grad_lambda2
#         lambda1 = np.clip(lambda1, 0, 1)
#         lambda2 = np.clip(lambda2, 0, 1)
#         if it % 50 == 0:
#             obj = hierarchical_lda_objective(W, S_B, S_W, subclass_means, parent_means, lambda1, lambda2)
#             print(f"Iter {it}: Obj={obj:.4f}, lambda1={lambda1:.4f}, lambda2={lambda2:.4f}")
#     return W, lambda1, lambda2

# W_opt, best_l1, best_l2 = joint_gradient_ascent(S_B, S_W, subclass_means, parent_means, reg, num_iters=5000)
# S_W_best = S_W  # S_W remains unchanged as computed earlier
# print("Best Obj:", hierarchical_lda_objective(W_opt, S_B, S_W_best, subclass_means, parent_means, best_l1, best_l2))
# print("Best lambda1:", best_l1)
# print("Best lambda2:", best_l2)

# # ----------------------
# # 5. Project the Data and Plot the Results Using the Best W
# # ----------------------
# data_points_2d = data_points @ W_opt
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# fig.tight_layout(pad=4.0)
# num_classes = len(unique_classes)
# discrete_cmap_all = plt.cm.get_cmap('viridis', num_classes)
# ax = axs[0, 0]
# scatter_all = ax.scatter(data_points_2d[:, 0], data_points_2d[:, 1], c=labels_class, cmap=discrete_cmap_all, alpha=0.7)
# ax.set_xlabel("Component 1")
# ax.set_ylabel("Component 2")
# ax.set_title("Hierarchical LDA Projection: All Classes")
# cbar_all = fig.colorbar(scatter_all, ax=ax, ticks=unique_classes)
# cbar_all.set_label('Class Label')
# cbar_all.ax.set_yticklabels(unique_classes)
# positions = [(0, 1), (1, 0), (1, 1)]
# for pos, c in zip(positions, unique_classes):
#     i, j = pos
#     ax = axs[i, j]
#     indices = np.where(labels_class == c)[0]
#     data_class_2d = data_points_2d[indices, :]
#     subclass_vals = np.array([lbl[1] for lbl in labels_cluster[indices]])
#     unique_subclasses = np.unique(subclass_vals)
#     discrete_cmap_sub = plt.cm.get_cmap('plasma', len(unique_subclasses))
#     scatter_sub = ax.scatter(data_class_2d[:, 0], data_class_2d[:, 1], c=subclass_vals, cmap=discrete_cmap_sub, alpha=0.7)
#     ax.set_xlabel("Component 1")
#     ax.set_ylabel("Component 2")
#     ax.set_title(f"Class {c} (colored by subclass)")
#     cbar_sub = fig.colorbar(scatter_sub, ax=ax, ticks=unique_subclasses)
#     cbar_sub.set_label("Subclass (cluster index)")
#     cbar_sub.ax.set_yticklabels(unique_subclasses)
# plt.suptitle(f"Optimized Hierarchical LDA Projection\n(Best lambda1={best_l1:.2f}, lambda2={best_l2:.2f})", fontsize=14)
# plt.tight_layout()
# plt.show()

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
S_W = np.zeros((d, d))
for c in unique_classes:
    idx = np.where(labels_class == c)[0]
    X_c = data_points[idx, :]
    mu_c = np.mean(X_c, axis=0)
    diff = X_c - mu_c
    S_W += diff.T @ diff
reg = 1e-8
S_W += reg * np.eye(d)
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
        mu_cs = np.mean(X_cs, axis=0)
        subclass_means[c].append(mu_cs)

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
# 4. Joint Gradient Ascent on W, lambda1, and lambda2 with Gradient History
# ----------------------
def joint_gradient_ascent(S_B, S_W, subclass_means, parent_means, reg, num_iters=500, step_W=1e-4, step_l1=1e-4, step_l2=1e-4, eps=1e-8):
    W = np.random.randn(d, 2)
    W, _ = np.linalg.qr(W)
    lambda1 = 0.5
    lambda2 = 0.5
    obj_history = []
    gradW_history = []
    lambda1_history = []
    lambda2_history = []
    for it in range(num_iters):
        f = np.trace(W.T @ S_B @ W)
        g = np.trace(W.T @ S_W @ W) + eps
        grad_W = (g * (2 * S_B @ W) - f * (2 * S_W @ W)) / (g**2)
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
        lambda1 = np.clip(lambda1, 0, 1)
        lambda2 = np.clip(lambda2, 0, 1)
        obj = hierarchical_lda_objective(W, S_B, S_W, subclass_means, parent_means, lambda1, lambda2)
        obj_history.append(obj)
        gradW_history.append(np.linalg.norm(grad_W))
        lambda1_history.append(lambda1)
        lambda2_history.append(lambda2)
        if it % 50 == 0:
            print(f"Iter {it}: Obj={obj:.4f}, lambda1={lambda1:.4f}, lambda2={lambda2:.4f}")
    return W, lambda1, lambda2, obj_history, gradW_history, lambda1_history, lambda2_history

W_opt, best_l1, best_l2, obj_history, gradW_history, lambda1_history, lambda2_history = joint_gradient_ascent(S_B, S_W, subclass_means, parent_means, reg, num_iters=5000)
print("Best Obj:", hierarchical_lda_objective(W_opt, S_B, S_W, subclass_means, parent_means, best_l1, best_l2))
print("Best lambda1:", best_l1)
print("Best lambda2:", best_l2)

# ----------------------
# 5. Plot Gradient History and Final Projection
# ----------------------
fig_hist, axs_hist = plt.subplots(3, 1, figsize=(8, 12))
axs_hist[0].plot(obj_history)
axs_hist[0].set_xlabel("Iteration")
axs_hist[0].set_ylabel("Objective")
axs_hist[0].set_title("Objective History")
axs_hist[1].plot(gradW_history, color='orange')
axs_hist[1].set_xlabel("Iteration")
axs_hist[1].set_ylabel("Grad W Norm")
axs_hist[1].set_title("W Gradient Norm History")
axs_hist[2].plot(lambda1_history, label='lambda1', color='green')
axs_hist[2].plot(lambda2_history, label='lambda2', color='red')
axs_hist[2].set_xlabel("Iteration")
axs_hist[2].set_ylabel("Lambda Value")
axs_hist[2].set_title("Lambda History")
axs_hist[2].legend()
plt.tight_layout()
plt.show()

data_points_2d = data_points @ W_opt
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
plt.suptitle(f"Optimized Hierarchical LDA Projection\n(Best lambda1={best_l1:.2f}, lambda2={best_l2:.2f})", fontsize=14)
plt.tight_layout()
plt.show()
