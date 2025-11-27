import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import KFold
import itertools
import time
import os

# ==========================================
# 1. Data Generation
# ==========================================
def generate_data(dims=1000):
    np.random.seed(42)
    clusters_per_class = {1: [300], 2: [150, 150], 3: [75, 75, 75, 75]}
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
            
    return np.vstack(data_points), np.array(labels_class), np.array(labels_cluster)

# ==========================================
# 2. Matrices & Helper Functions
# ==========================================
def compute_matrices(data, labels_class, labels_cluster, dims):
    overall_mean = np.mean(data, axis=0)
    unique_classes = np.unique(labels_class)
    S_B = np.zeros((dims, dims)); S_WS = np.zeros((dims, dims)); S_BS = np.zeros((dims, dims))
    parent_means = {}; subclass_means = {}
    
    for c in unique_classes:
        idx = np.where(labels_class == c)[0]
        X_c = data[idx, :]; N_c = X_c.shape[0]; mu_c = np.mean(X_c, axis=0)
        parent_means[c] = mu_c
        S_B += N_c * np.outer(mu_c - overall_mean, mu_c - overall_mean)
        
        # Extract subclass labels for this parent class
        class_subs = np.array([lbl[1] for lbl in labels_cluster[idx]])
        subclass_means[c] = []
        for sub in np.unique(class_subs):
            mask = np.array([(lbl[0] == c and lbl[1] == sub) for lbl in labels_cluster])
            idx_sub = np.where(mask)[0]; X_cs = data[idx_sub, :]
            N_cs = X_cs.shape[0]; mu_cs = np.mean(X_cs, axis=0)
            subclass_means[c].append(mu_cs)
            diff = X_cs - mu_cs
            S_WS += diff.T @ diff
            S_BS += N_cs * np.outer(mu_cs - mu_c, mu_cs - mu_c)
    return S_B, S_WS, S_BS, parent_means, subclass_means

def grad_norm_wrt_W(W, diff_vec, eps=1e-8):
    """Computes gradient and value of ||W^T @ diff_vec||_2"""
    proj = W.T @ diff_vec
    norm_val = np.linalg.norm(proj) + eps
    grad = np.outer(diff_vec, proj) / norm_val
    return grad, norm_val

# ==========================================
# 3. Optimization Logic (Version 1 - Ratio Trace)
# ==========================================
def joint_gradient_ascent_v1(S_B, S_WS, S_BS, subclass_means, parent_means, reg=1e-4, num_iters=300, fix_hyperparams=None):
    dims = S_B.shape[0]
    W = np.random.randn(dims, 2); W, _ = np.linalg.qr(W)
    
    # Defaults or Fixed Hyperparameters
    alpha = fix_hyperparams['alpha'] if fix_hyperparams else 0.5
    l1 = fix_hyperparams['l1'] if fix_hyperparams else 0.5
    l2 = fix_hyperparams['l2'] if fix_hyperparams else 0.5
    
    eps = 1e-8; step_W = 1e-4; step_hyp = 1e-3
    history = []; hW = []
    
    for it in range(num_iters):
        S_W = alpha * S_WS + (1 - alpha) * S_BS + reg * np.eye(dims)
        
        # --- Ratio Trace Gradient Logic ---
        # J = Tr(D^-1 N)
        N_mat = W.T @ S_B @ W
        D_mat = W.T @ S_W @ W + eps * np.eye(2)
        D_inv = np.linalg.inv(D_mat)
        
        # Objective Value (LDA Part)
        # Using linalg.solve is more stable than D_inv @ N_mat for value, but we need D_inv for gradient
        lda_val = np.trace(np.linalg.solve(D_mat, N_mat))
        
        # Gradient W (LDA Part)
        # Grad = 2 * Sb * W * D^-1 - 2 * Sw * W * (D^-1 * N * D^-1)
        term1 = 2 * S_B @ W @ D_inv
        K = D_inv @ N_mat @ D_inv
        term2 = 2 * S_W @ W @ K
        grad_W = term1 - term2
        
        # Gradient Alpha
        # Grad_Alpha = -Tr( W' (dSw/da) W K )
        grad_alpha = -np.trace(W.T @ (S_WS - S_BS) @ W @ K)
        
        # --- Regularizers (Additive in V1) ---
        
        # R1 (Separation: Sum 1/||diff||)
        g_R1 = np.zeros_like(W)
        R1_val = 0.0
        for c, means in subclass_means.items():
            num_sub = len(means)
            for i in range(num_sub):
                for j in range(i+1, num_sub):
                    diff = means[i] - means[j]
                    g_norm, n_val = grad_norm_wrt_W(W, diff)
                    R1_val += 1.0 / (n_val + eps)
                    # d(1/x)/dW = -1/x^2 * dx/dW
                    g_R1 += -(1 / (n_val**2)) * g_norm
                    
        # R2 (Collapse: Sum ||diff||) - Standard linear norm, no hinge
        g_R2 = np.zeros_like(W)
        R2_val = 0.0
        for c, means in subclass_means.items():
            for mu in means:
                diff = mu - parent_means[c]
                g_norm, n_val = grad_norm_wrt_W(W, diff)
                R2_val += n_val
                g_R2 += g_norm

        # Total Gradient
        # J_total = LDA_Obj + l1 * R1 + l2 * R2
        total_grad_W = grad_W + l1 * g_R1 + l2 * g_R2
        
        # Update Hyperparams if not fixed
        if not fix_hyperparams:
            # Gradient ascent on params to maximize J
            alpha = np.clip(alpha + step_hyp * grad_alpha, 0, 1)
            l1 = max(l1 + step_hyp * R1_val, 0) # R1 is positive, increasing l1 increases J
            l2 = max(l2 + step_hyp * R2_val, 0) # R2 is positive, increasing l2 increases J
            
        W += step_W * total_grad_W
        W, _ = np.linalg.qr(W)
        
        # Record total objective
        obj = lda_val + l1 * R1_val + l2 * R2_val
        history.append(obj)
        hW.append(np.linalg.norm(total_grad_W))
        
        if it % 50 == 0:
            print(f"Iter {it}: Obj={obj:.4f}, a={alpha:.2f}, l1={l1:.2f}, l2={l2:.2f}", end='\r')
            
    return W, alpha, l1, l2, history, hW

# ==========================================
# 4. Grid Search Wrapper
# ==========================================
def run_grid_search(S_B, S_WS, S_BS, sub_means, par_means, reg):
    print("Running Grid Search V1 (Ratio Trace)...")
    alphas = np.linspace(0, 1, 3)
    l1s = np.linspace(0.1, 10, 3)
    l2s = np.linspace(0.1, 10, 3)
    
    grid = {'alpha': alphas, 'l1': l1s, 'l2': l2s}
    keys, values = zip(*grid.items())
    
    best_obj = -np.inf
    best_params = None
    best_W = None
    
    total_combos = np.prod([len(v) for v in values])
    print(f"Total Combinations: {total_combos}")
    
    count = 0
    for bundle in itertools.product(*values):
        params = dict(zip(keys, bundle))
        # Short optimization
        W_curr, _, _, _, hist, _ = joint_gradient_ascent_v1(
            S_B, S_WS, S_BS, sub_means, par_means, 
            reg=reg, num_iters=30, fix_hyperparams=params
        )
        if hist[-1] > best_obj:
            best_obj = hist[-1]
            best_params = params
            best_W = W_curr.copy()
            
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{total_combos} | Best Obj: {best_obj:.2f}", end='\r')
            
    print(f"\nGrid Search Complete. Best Obj: {best_obj:.4f}")
    return best_params, best_W

# ==========================================
# 5. Visualizations
# ==========================================
def run_visualizations(W, alpha, l1, l2, data, y_cls, y_clst, hist, hist_W, dims, title_prefix):
    # 1. Full Projection
    proj = data @ W
    u_parents = np.unique(y_cls)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Overall Scatter
    cmap_parents = cm.get_cmap('viridis', len(u_parents))
    for i, c in enumerate(u_parents):
        idx = np.where(y_cls == c)[0]
        axs[0].scatter(proj[idx, 0], proj[idx, 1], label=f'Parent {c}', 
                       color=cmap_parents(i), alpha=0.7, s=40)
        
    axs[0].set_title(f"{title_prefix} Projection\n(a={alpha:.2f}, l1={l1:.2f}, l2={l2:.2f})")
    axs[0].set_xlabel("Component 1")
    axs[0].set_ylabel("Component 2")
    axs[0].legend()
    axs[0].grid(True, alpha=0.2)
    
    # Gradient History
    if hist_W and len(hist_W) > 0:
        axs[1].plot(hist_W, label='Grad W Norm', color='purple')
        axs[1].set_xscale('log') 
        axs[1].set_yscale('log')
        axs[1].set_title("Gradient Norm History")
        axs[1].set_xlabel("Iteration")
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "No History for Grid Search Snapshot", ha='center')
    plt.tight_layout()
    save_path = 'figs/hlda_v1'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{title_prefix}_projection.png"))

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    dims = 1000
    reg = 1e-4
    
    print("1. Generating Data...")
    data, y_cls, y_clst = generate_data(dims)
    
    print("2. Computing Scatter Matrices...")
    Sb, Sws, Sbs, pm, sm = compute_matrices(data, y_cls, y_clst, dims)
    
    # --- PHASE A: GRID SEARCH ---
    best_p, best_W_grid = run_grid_search(Sb, Sws, Sbs, sm, pm, reg)
    
    # Visualize Grid Result
    print("\nVisualizing Best Grid Search Result...")
    run_visualizations(best_W_grid, best_p['alpha'], best_p['l1'], best_p['l2'],
                       data, y_cls, y_clst, [], [], dims, title_prefix="Grid Search Best")
    
    # --- PHASE B: GRADIENT ASCENT ---
    print("3. Running Full Gradient Ascent using Best Grid Params...")
    W, a, l1, l2, h, hW = joint_gradient_ascent_v1(
        Sb, Sws, Sbs, sm, pm, 
        reg=reg, num_iters=1000, fix_hyperparams=best_p
    )
    
    print(f"Final Result: Obj={h[-1]:.4f}")
    run_visualizations(W, a, l1, l2, data, y_cls, y_clst, h, hW, dims, title_prefix="Final Gradient Ascent")