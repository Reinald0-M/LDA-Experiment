import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import matplotlib
# ----------------------
# 1. Data Generation (Hierarchical Data)
# ----------------------
np.random.seed(42)  # Set seed for reproducibility
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
# 2. Compute Scatter Matrices and Means
# ----------------------
def compute_scatter_matrices(data, labels_class, labels_cluster, dims):
    overall_mean = np.mean(data, axis=0)
    unique_classes = np.unique(labels_class)
    
    # Global Between-Class Scatter (S_B)
    S_B = np.zeros((dims, dims))
    for c in unique_classes:
        idx = np.where(labels_class == c)[0]
        X_c = data[idx, :]
        N_c = X_c.shape[0]
        mu_c = np.mean(X_c, axis=0)
        S_B += N_c * np.outer(mu_c - overall_mean, mu_c - overall_mean)
    
    S_WS = np.zeros((dims, dims))
    S_BS = np.zeros((dims, dims))
    parent_means = {}
    subclass_means = {}
    
    # We need the global mean (mu_tot) for the Tau regularization term
    mu_tot = overall_mean 
    
    for c in unique_classes:
        idx = np.where(labels_class == c)[0]
        X_c = data[idx, :]
        mu_c = np.mean(X_c, axis=0)
        parent_means[c] = mu_c
        
        # Extract subclasses for this class
        class_subs = np.array([lbl[1] for lbl in labels_cluster[idx]])
        unique_subs = np.unique(class_subs)
        subclass_means[c] = []
        
        for sub in unique_subs:
            # Filter specifically for (Class=c, Cluster=sub)
            mask = np.array([(lbl[0] == c and lbl[1] == sub) for lbl in labels_cluster])
            idx_sub = np.where(mask)[0]
            
            X_cs = data[idx_sub, :]
            N_cs = X_cs.shape[0]
            mu_cs = np.mean(X_cs, axis=0)
            subclass_means[c].append(mu_cs)
            
            diff = X_cs - mu_cs
            S_WS += diff.T @ diff
            S_BS += N_cs * np.outer(mu_cs - mu_c, mu_cs - mu_c)
            
    return S_B, S_WS, S_BS, parent_means, subclass_means, mu_tot

# ----------------------
# 3. Gradient Helper: Derivative of Norms
# ----------------------
def grad_norm_wrt_W(W, diff_vec, eps=1e-8):
    """
    Computes gradient of ||W^T @ diff_vec||_2 w.r.t W.
    Derivative is: (diff_vec @ (diff_vec^T W)) / ||W^T diff_vec||
    """
    proj = W.T @ diff_vec
    norm_val = np.linalg.norm(proj) + eps
    
    # Gradient: Outer product of input vector and projected vector, scaled by inverse norm
    grad_term = np.outer(diff_vec, proj) / norm_val
    return grad_term, norm_val

# ----------------------
# 4. Joint Gradient Ascent (Ratio Objective + Hinge Loss)
# ----------------------
def joint_gradient_ascent_ratio(S_B, S_WS, S_BS, subclass_means, parent_means, mu_tot, 
                                num_iters=500, eps=1e-8):
    
    dims = S_B.shape[0]
    
    # Initialize W on Stiefel Manifold (Orthonormal)
    W = np.random.randn(dims, 2)
    W, _ = np.linalg.qr(W) 
    
    # Hyperparameters
    lambda1 = 0.5 
    lambda2 = 0.5
    alpha = 0.5
    tau = 0.5
    reg = 1e-4  # Fixed: Defined 'reg' locally for diagonal loading
    
    # Steps / Learning Rates
    step_W = 1e-4
    step_l = 1e-3
    step_alpha = 1e-4
    step_tau = 1e-4

    history = []
    
    for it in range(num_iters):
        # 1. Compute S_mix (Denominator Scatter)
        # S_mix = alpha * S_WS + (1-alpha) * S_BS
        S_mix = alpha * S_WS + (1 - alpha) * S_BS + reg * np.eye(dims)
        
        # 2. Compute S_R (Numerator Regularizer: Sibling Separation)
        # Formula: Sum 1 / (||W^T(mu_a - mu_b)||)
        S_R_val = 0.0
        grad_S_R = np.zeros_like(W)
        
        for c, sub_means in subclass_means.items():
            num_sub = len(sub_means)
            for i in range(num_sub):
                for j in range(i+1, num_sub):
                    diff = sub_means[i] - sub_means[j]
                    
                    # Chain rule: d(1/x)/dW = -1/x^2 * dx/dW
                    grad_norm, norm_val = grad_norm_wrt_W(W, diff)
                    inv_norm = 1.0 / (norm_val + eps)
                    
                    S_R_val += inv_norm
                    grad_S_R += -(inv_norm**2) * grad_norm

        # 3. Compute S_T (Denominator Regularizer: Topology Hinge Loss)
        # Formula: Sum max(0, ||Anchor|| - tau * ||Global||)
        S_T_val = 0.0
        grad_S_T = np.zeros_like(W)
        grad_S_T_tau_part = 0.0 # Tracks sum of ||Global|| for active constraints
        
        for c, sub_means in subclass_means.items():
            for mu_cs in sub_means:
                # Part A: Anchor (subclass -> parent)
                diff_anchor = mu_cs - parent_means[c]
                g_anchor, n_anchor = grad_norm_wrt_W(W, diff_anchor)
                
                # Part B: Global Separation (subclass -> global mean)
                diff_global = mu_cs - mu_tot
                g_global, n_global = grad_norm_wrt_W(W, diff_global)
                
                # --- HINGE LOSS LOGIC ---
                # Constraint: n_anchor < tau * n_global
                # Loss is positive only if constraint is violated
                loss = n_anchor - tau * n_global
                
                if loss > 0:
                    # Constraint Violated: Add penalty and gradients
                    S_T_val += loss
                    
                    # Gradient w.r.t W
                    grad_S_T += (g_anchor - tau * g_global)
                    
                    # Gradient w.r.t Tau component (derivative of -tau*G is -G)
                    # We store just G here to handle the chain rule cleanly later
                    grad_S_T_tau_part += n_global

        # 4. Numerator (N) and Denominator (D)
        tr_num = np.trace(W.T @ S_B @ W)
        tr_den = np.trace(W.T @ S_mix @ W)
        
        N = tr_num + lambda2 * S_R_val
        D = tr_den + lambda1 * S_T_val + eps 
        
        obj = N / D
        
        # 5. Gradients via Quotient Rule
        # J = N / D  =>  grad = (D * grad_N - N * grad_D) / D^2
        
        # Gradients wrt W
        grad_N_W = 2 * S_B @ W + lambda2 * grad_S_R
        grad_D_W = 2 * S_mix @ W + lambda1 * grad_S_T
        grad_J_W = (D * grad_N_W - N * grad_D_W) / (D**2)
        
        # Gradients wrt Hyperparameters
        
        # lambda2 (only in N): dJ/dl2 = S_R / D
        grad_l2 = S_R_val / D
        
        # lambda1 (only in D): dJ/dl1 = -(N * S_T) / D^2
        grad_l1 = -(N * S_T_val) / (D**2)
        
        # alpha (only in D): S_mix = a*WS + (1-a)*BS
        # d(S_mix)/da = S_WS - S_BS
        dD_da = np.trace(W.T @ (S_WS - S_BS) @ W)
        grad_alpha = -(N * dD_da) / (D**2)
        
        # tau (only in D via S_T):
        # d(S_T)/dtau = -1 * sum(n_global_active) = -grad_S_T_tau_part
        # dD/dtau = lambda1 * (-grad_S_T_tau_part)
        dD_dtau = - lambda1 * grad_S_T_tau_part
        grad_tau = -(N * dD_dtau) / (D**2)

        # 6. Parameter Updates (Gradient Ascent)
        W += step_W * grad_J_W
        W, _ = np.linalg.qr(W) # Retract to Stiefel manifold
        
        lambda1 += step_l * grad_l1
        lambda2 += step_l * grad_l2
        alpha += step_alpha * grad_alpha
        tau += step_tau * grad_tau
        
        # 7. Projections / Constraints
        lambda1 = max(lambda1, 0.0)
        lambda2 = max(lambda2, 0.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        tau = np.clip(tau, 0.0, 1.0)
        
        history.append(obj)
        
        if it % 50 == 0:
            print(f"Iter {it}: Obj={obj:.4f}, a={alpha:.2f}, tau={tau:.2f}, l1={lambda1:.2f}, l2={lambda2:.2f}")

    # Fixed: Removed assignment expressions (:=) from return
    return W, alpha, tau, lambda1, lambda2, history

# ----------------------
# 5. Main Execution
# ----------------------
if __name__ == "__main__":
    # Compute Matrices
    S_B_full, S_WS_full, S_BS_full, parent_means, subclass_means, mu_tot = compute_scatter_matrices(
        data_points, labels_class, labels_cluster, dims
    )

    print("Starting Joint Gradient Ascent with Ratio Objective (Hinge Loss)...")
    
    # Run Optimization
    W_opt, best_alpha, best_tau, best_l1, best_l2, history = joint_gradient_ascent_ratio(
        S_B_full, S_WS_full, S_BS_full, subclass_means, parent_means, mu_tot, num_iters=5000
    )

    print(f"\nFinal Results:")
    print(f"Best Objective: {history[-1]:.4f}")
    print(f"Alpha: {best_alpha:.4f}")
    print(f"Tau: {best_tau:.4f}")
    print(f"Lambda1: {best_l1:.4f}")
    print(f"Lambda2: {best_l2:.4f}")
# ----------------------
    # 6. Plotting Results
    # ----------------------
    import matplotlib as mpl # Ensure this is imported

    # Plot History
    plt.figure(figsize=(10,4))
    plt.plot(history)
    plt.title("Ratio Objective J*(W) over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot Projection
    data_2d = data_points @ W_opt
    plt.figure(figsize=(8, 6))
    
    # --- FIXED COLORMAP LINE ---
    # Get the colormap object first, then resample it to the number of classes
    n_classes = len(np.unique(labels_class))
    discrete_cmap = mpl.colormaps['viridis'].resampled(n_classes) 
    
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_class, cmap=discrete_cmap, alpha=0.6)
    plt.title(f"Projected Data (Ratio Method)\nalpha={best_alpha:.2f}, tau={best_tau:.2f}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Class")
    plt.show()