import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import KFold
import itertools
import time
import os
# v2 of Hierarchical LDA with Ratio Trace Optimization
# ----------------------
# 1. Data Generation
# ----------------------
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

# ----------------------
# 2. Matrices & Helper
# ----------------------
def compute_matrices(data, labels_class, labels_cluster, dims):
    overall_mean = np.mean(data, axis=0)
    unique_classes = np.unique(labels_class)
    S_B = np.zeros((dims, dims)); S_WS = np.zeros((dims, dims)); S_BS = np.zeros((dims, dims))
    parent_means = {}; subclass_means = {}; mu_tot = overall_mean
    
    for c in unique_classes:
        idx = np.where(labels_class == c)[0]
        X_c = data[idx, :]; N_c = X_c.shape[0]; mu_c = np.mean(X_c, axis=0)
        parent_means[c] = mu_c
        S_B += N_c * np.outer(mu_c - overall_mean, mu_c - overall_mean)
        
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
    return S_B, S_WS, S_BS, parent_means, subclass_means, mu_tot

# ----------------------
# 3. Optimization V2 (Ratio Trace)
# ----------------------
def joint_gradient_ascent(S_B, S_WS, S_BS, subclass_means, parent_means, mu_tot, reg, num_iters=500, fix_params=None):
    dims = S_B.shape[0]
    W = np.random.randn(dims, 2)
    W, _ = np.linalg.qr(W)
    
    # Defaults
    l1 = fix_params['l1'] if fix_params else 0.5
    l2 = fix_params['l2'] if fix_params else 0.5
    alpha = fix_params['alpha'] if fix_params else 0.5
    tau = fix_params['tau'] if fix_params else 0.5
    
    step=1e-4; eps=1e-8
    history = []; hW = []
    
    for it in range(num_iters):
        S_W = alpha * S_WS + (1 - alpha) * S_BS + reg * np.eye(dims)
        
        # --- Ratio Trace Logic ---
        # J = Tr( (W'SwW)^-1 (W'SbW) )
        N_mat = W.T @ S_B @ W
        D_mat = W.T @ S_W @ W + eps * np.eye(2)
        D_inv = np.linalg.inv(D_mat)
        
        # Objective Value
        lda_val = np.trace(D_inv @ N_mat)
        
        # Gradients
        # Grad W = 2*Sb*W*D^-1 - 2*Sw*W*(D^-1 N D^-1)
        term1 = 2 * S_B @ W @ D_inv
        K = D_inv @ N_mat @ D_inv # Intermediate term
        term2 = 2 * S_W @ W @ K
        grad_W = term1 - term2
        
        # Grad Alpha = -Tr( W' (dSw/da) W K )
        grad_alpha = -np.trace(W.T @ (S_WS - S_BS) @ W @ K)
        
        # --- Regularizers ---
        # R1 (Standard Separation)
        g_R1 = np.zeros_like(W)
        for c, means in subclass_means.items():
            for i in range(len(means)):
                for j in range(i+1, len(means)):
                    diff = means[i] - means[j]; proj = W.T @ diff
                    n_val = np.linalg.norm(proj) + eps
                    g_R1 += -(1/n_val**2) * (np.outer(diff, proj)/n_val)
                    
        # R2 (Hinge)
        g_R2 = np.zeros_like(W)
        for c, means in subclass_means.items():
            for mu in means:
                d_anc = mu - parent_means[c]; p_anc = W.T @ d_anc; n_anc = np.linalg.norm(p_anc) + eps
                d_glo = mu - mu_tot; p_glo = W.T @ d_glo; n_glo_sq = np.dot(p_glo, p_glo)
                
                if (n_anc - tau * n_glo_sq) > 0:
                    g_anc = np.outer(d_anc, p_anc)/n_anc
                    g_glo_sq = 2 * np.outer(d_glo, p_glo)
                    g_R2 += (g_anc - tau * g_glo_sq)
        
        # Total Gradient
        W += step * (grad_W + l1*g_R1 + l2*g_R2)
        W, _ = np.linalg.qr(W)
        
        # Hyperparameter Updates
        if not fix_params:
            alpha = np.clip(alpha + step * grad_alpha, 0, 1)
            # Tau update heuristic (Approx descent)
            # hA.append(np.linalg.norm(grad_alpha))
            
        history.append(lda_val)
        hW.append(np.linalg.norm(grad_W))
        
    return W, l1, l2, alpha, tau, history, hW

# ----------------------
# 4. Grid Search
# ----------------------
def run_grid(S_B, S_WS, S_BS, sm, pm, mt, reg):
    print("Running Grid Search V2 (Ratio Trace)...") 
    alphas = np.linspace(0, 1, 2)
    l1s = np.linspace(0.1, 1000, 2)
    l2s = np.linspace(0.1, 1000, 2)
    taus = np.linspace(0, 1, 2)
    
    grid = {'alpha': alphas, 'l1': l1s, 'l2': l2s, 'tau': taus}
    keys, values = zip(*grid.items())
    
    best_obj = -np.inf
    best_params = None
    best_W = None
    
    total_combos = np.prod([len(v) for v in values])
    print(f"Total Combinations: {total_combos}")
    
    count = 0
    start_time = time.time()
    
    for bundle in itertools.product(*values):
        params = dict(zip(keys, bundle))
        W_curr,_,_,_,_, hist, _, = joint_gradient_ascent(S_B, S_WS, S_BS, sm, pm, mt, reg, 100, params)
        
        if hist[-1] > best_obj: 
            best_obj = hist[-1]
            best_params = params
            best_W = W_curr.copy()
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{total_combos} | Best Obj: {best_obj:.2f}", end='\r')
    print(f"\nGrid Search Complete. Best Obj: {best_obj:.4f}")
    return best_params, best_W

# ----------------------
# 5. Visualization
# ----------------------
def viz(W, a, l1, l2, t, data, y_cls, y_clst, hist_W, dims, title_prefix="V2"):
    # Full Projection
    d2 = data @ W
    u = np.unique(y_cls)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # 1. Overall Scatter
    sc = axs[0].scatter(d2[:,0], d2[:,1], c=y_cls, cmap='viridis', alpha=0.7)
    axs[0].set_title(f"{title_prefix} Projection\n(a={a:.2f}, l1={l1:.2f}, l2={l2:.2f}, t={t:.2f})")
    axs[0].set_xlabel("Component 1")
    axs[0].set_ylabel("Component 2")
    plt.colorbar(sc, ax=axs[0], label='Parent Class')
    
    # 2. Gradient History (if available)
    if hist_W and len(hist_W) > 0:
        axs[1].plot(hist_W, label='Grad W Norm', color='purple')
        axs[1].set_yscale('log')
        axs[1].set_xscale('log')
        axs[1].set_title("Gradient Norm History")
        axs[1].set_xlabel("Iteration")
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "No History for Grid Search Snapshot", ha='center')
    
    plt.tight_layout()
    save_path = 'figs/hlda_v2'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{title_prefix}_projection.png"))

if __name__ == "__main__":
    dims = 100
    reg = 1e-8
    
    print("1. Generating Data...")
    d, yc, ycl = generate_data(dims)
    
    print("2. Computing Matrices...")
    Sb, Sw, Sbs, pm, sm, mt = compute_matrices(d, yc, ycl, dims)
    
    # --- PHASE A: GRID SEARCH ---
    best_p, best_W_grid = run_grid(Sb, Sw, Sbs, sm, pm, mt, reg)
    
    # Visualize Grid Result
    print("\nVisualizing Best Grid Search Result...")
    viz(best_W_grid, best_p['alpha'], best_p['l1'], best_p['l2'], best_p['tau'], 
        d, yc, ycl, [], dims, title_prefix="Grid Search Best")
    
    # --- PHASE B: GRADIENT ASCENT ---
    # Use best params from grid, run longer optimization
    print("3. Running Full Gradient Ascent using Best Grid Params...")
    W_final, l1, l2, a, t, h, hW = joint_gradient_ascent(
        Sb, Sw, Sbs, sm, pm, mt, reg, 
        num_iters=1000, 
        fix_params=best_p
    )
    
    # Visualize Final Result
    print(f"Final Objective: {h[-1]:.4f}")
    viz(W_final, a, l1, l2, t, d, yc, ycl, hW, dims, title_prefix="Final Gradient Ascent")