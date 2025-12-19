import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import itertools
import time
import os

# # ==========================================
# # 1. Real World Data Loading (Fashion MNIST)
# # ==========================================
# def get_fashion_mnist_hierarchy(max_samples=3000):
#     """
#     Fetches Fashion-MNIST and structures it into a 2-level hierarchy.
#     Native Dim: 784 (28x28 images)
#     """
#     print("Fetching Fashion-MNIST from OpenML (this may take a moment)...")
#     X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, cache=True)
#     y = y.astype(int)
    
#     # Hierarchy Mapping
#     # 1: Footwear (Sandal, Sneaker, Ankle Boot)
#     # 2: Tops/Outer (T-shirt, Pullover, Coat, Shirt)
#     # 3: Legs/Body (Trouser, Dress)
#     hierarchy_map = {
#         1: [5, 7, 9],       
#         2: [0, 2, 4, 6],    
#         3: [1, 3]           
#     }
    
#     # Re-indexing Logic
#     subclass_reindex_map = {} 
#     for pid, subclasses in hierarchy_map.items():
#         for i, sub in enumerate(subclasses):
#             subclass_reindex_map[sub] = (pid, i + 1)
            
#     # Filter
#     valid_mask = np.isin(y, [c for sublist in hierarchy_map.values() for c in sublist])
#     X_filtered = X[valid_mask]
#     y_filtered = y[valid_mask]
    
#     # Downsample for speed
#     if max_samples and max_samples < len(X_filtered):
#         rng = np.random.RandomState(42)
#         indices = rng.permutation(len(X_filtered))[:max_samples]
#         X_filtered = X_filtered[indices]
#         y_filtered = y_filtered[indices]
    
#     final_labels_class = []
#     final_labels_cluster = []
    
#     for label in y_filtered:
#         pid, sub_idx = subclass_reindex_map[label]
#         final_labels_class.append(pid)
#         final_labels_cluster.append((pid, sub_idx))
        
#     # Standardize
#     scaler = StandardScaler()
#     X_norm = scaler.fit_transform(X_filtered)
    
#     print(f"Data Loaded: {X_norm.shape[0]} samples, {X_norm.shape[1]} dimensions.")
#     return X_norm, np.array(final_labels_class), np.array(final_labels_cluster)

# # ==========================================
# # 2. Matrices & Helper Functions
# # ==========================================
# def compute_matrices(data, labels_class, labels_cluster, dims):
#     overall_mean = np.mean(data, axis=0)
#     unique_classes = np.unique(labels_class)
#     S_B = np.zeros((dims, dims)); S_WS = np.zeros((dims, dims)); S_BS = np.zeros((dims, dims))
#     parent_means = {}; subclass_means = {}
#     mu_tot = overall_mean
    
#     for c in unique_classes:
#         idx = np.where(labels_class == c)[0]
#         X_c = data[idx, :]; N_c = X_c.shape[0]; mu_c = np.mean(X_c, axis=0)
#         parent_means[c] = mu_c
#         S_B += N_c * np.outer(mu_c - overall_mean, mu_c - overall_mean)
        
#         # Filter tuples for current parent class
#         current_cluster_labels = labels_cluster[idx]
#         unique_subs = np.unique([lbl[1] for lbl in current_cluster_labels])
        
#         subclass_means[c] = []
#         for sub in unique_subs:
#             # Mask: Parent == c AND SubIndex == sub
#             mask = np.array([(lbl[0] == c and lbl[1] == sub) for lbl in labels_cluster])
#             idx_sub = np.where(mask)[0]
            
#             if len(idx_sub) == 0: continue
            
#             X_cs = data[idx_sub, :]
#             N_cs = X_cs.shape[0]; mu_cs = np.mean(X_cs, axis=0)
#             subclass_means[c].append(mu_cs)
            
#             diff = X_cs - mu_cs
#             S_WS += diff.T @ diff
#             S_BS += N_cs * np.outer(mu_cs - mu_c, mu_cs - mu_c)
            
#     return S_B, S_WS, S_BS, parent_means, subclass_means, mu_tot

# def grad_norm_wrt_W(W, diff_vec, eps=1e-8):
#     """Computes gradient and value of ||W^T @ diff_vec||_2"""
#     proj = W.T @ diff_vec
#     norm_val = np.linalg.norm(proj) + eps
#     grad = np.outer(diff_vec, proj) / norm_val
#     return grad, norm_val

# # ==========================================
# # 3. Optimization Logic (Version 4 - Ratio + Squared Hinge)
# # ==========================================
# def joint_gradient_ascent_v4(S_B, S_WS, S_BS, subclass_means, parent_means, mu_tot, 
#                              reg=1e-4, num_iters=300, fix_hyperparams=None, verbose=True):
#     dims = S_B.shape[0]
#     # Seeding inside function for consistent starts across grid search
#     np.random.seed(42) 
#     W = np.random.randn(dims, 2); W, _ = np.linalg.qr(W)
    
#     # Defaults
#     alpha = fix_hyperparams['alpha'] if fix_hyperparams else 0.5
#     l1 = fix_hyperparams['l1'] if fix_hyperparams else 0.5
#     l2 = fix_hyperparams['l2'] if fix_hyperparams else 0.5
#     tau = fix_hyperparams['tau'] if fix_hyperparams else 0.8  # Fixed strictness
    
#     # Stability Constants
#     eps = 1e-8; step_W = 1e-4; step_hyp = 1e-3
#     gamma_alpha = 5.0   # Penalty for straying from 0.5
#     decay_l = 0.01      # Decay for l1/l2
    
#     history = []; hW = []
    
#     for it in range(num_iters):
#         # S_mix
#         S_mix = alpha * S_WS + (1 - alpha) * S_BS + reg * np.eye(dims)
        
#         # S_R (Numerator)
#         S_R_val = 0.0; grad_S_R = np.zeros_like(W)
#         for c, sub_means in subclass_means.items():
#             num_sub = len(sub_means)
#             for i in range(num_sub):
#                 for j in range(i+1, num_sub):
#                     diff = sub_means[i] - sub_means[j]
#                     grad, n_val = grad_norm_wrt_W(W, diff)
#                     inv_norm = 1.0 / (n_val + eps)
#                     S_R_val += inv_norm
#                     grad_S_R += -(inv_norm**2) * grad
                    
#         # S_T (Denominator)
#         S_T_val = 0.0; grad_S_T = np.zeros_like(W)
#         for c, sub_means in subclass_means.items():
#             for mu_cs in sub_means:
#                 diff_anchor = mu_cs - parent_means[c]
#                 g_anchor, n_anchor = grad_norm_wrt_W(W, diff_anchor)
                
#                 diff_global = mu_cs - mu_tot
#                 proj_global = W.T @ diff_global
#                 n_global_sq = np.dot(proj_global, proj_global)
#                 g_global_sq = 2 * np.outer(diff_global, proj_global)
                
#                 loss = n_anchor - tau * n_global_sq
                
#                 if loss > 0:
#                     S_T_val += loss
#                     grad_S_T += (g_anchor - tau * g_global_sq)

#         # Ratio & Penalties
#         tr_num = np.trace(W.T @ S_B @ W)
#         tr_den = np.trace(W.T @ S_mix @ W)
        
#         N = tr_num + l2 * S_R_val
#         D = tr_den + l1 * S_T_val + eps 
#         ratio = N / D
        
#         pen_alpha = gamma_alpha * (alpha - 0.5)**2
#         pen_l = decay_l * (l1**2 + l2**2)
        
#         # Gradients
#         grad_N_W = 2 * S_B @ W + l2 * grad_S_R
#         grad_D_W = 2 * S_mix @ W + l1 * grad_S_T
#         grad_J_W = (D * grad_N_W - N * grad_D_W) / (D**2)
        
#         if not fix_hyperparams:
#             dD_da = np.trace(W.T @ (S_WS - S_BS) @ W)
#             grad_alpha = (-(N * dD_da) / (D**2)) - (2 * gamma_alpha * (alpha - 0.5))
#             grad_l1 = (-(N * S_T_val) / (D**2)) - (2 * decay_l * l1)
#             grad_l2 = (S_R_val / D) - (2 * decay_l * l2)
            
#             l1 = max(l1 + step_hyp * grad_l1, 1e-4) 
#             l2 = max(l2 + step_hyp * grad_l2, 1e-4)
#             alpha = np.clip(alpha + step_hyp * grad_alpha, 0.01, 0.99)
            
#         W += step_W * grad_J_W; W, _ = np.linalg.qr(W)
#         history.append(ratio)
#         hW.append(np.linalg.norm(grad_J_W))
        
#         if verbose and it % 50 == 0:
#             print(f"Iter {it}: Ratio={ratio:.4f}, a={alpha:.2f}, l1={l1:.2f}, l2={l2:.2f} (tau={tau:.2f})", end='\r')
            
#     return W, alpha, l1, l2, tau, history, hW

# # ==========================================
# # 4. Grid Search Wrapper
# # ==========================================
# def run_grid_search(S_B, S_WS, S_BS, sub_means, par_means, mu_tot, reg):
#     print("Running Grid Search V4 (Ratio + Squared Hinge) on Fashion-MNIST...")
#     alphas = np.linspace(0, 1, 10)
#     l1s = np.linspace(0.1, 10, 10)
#     l2s = np.linspace(0.1, 10, 10)
#     taus = np.linspace(0, 1, 10) 
    
#     grid = {'alpha': alphas, 'l1': l1s, 'l2': l2s, 'tau': taus}
#     keys, values = zip(*grid.items())
    
#     best_obj = -np.inf
#     best_params = None
#     best_W = None
    
#     total_combos = np.prod([len(v) for v in values])
#     print(f"Total Combinations: {total_combos}")
    
#     count = 0
#     for bundle in itertools.product(*values):
#         params = dict(zip(keys, bundle))
        
#         # Use short iters for grid search
#         W_curr, _, _, _, _, hist, _ = joint_gradient_ascent_v4(
#             S_B, S_WS, S_BS, sub_means, par_means, mu_tot, 
#             reg=reg, num_iters=30, fix_hyperparams=params, 
#             verbose=False 
#         )
        
#         if hist[-1] > best_obj:
#             best_obj = hist[-1]
#             best_params = params
#             best_W = W_curr.copy()
            
#         count += 1
#         if count % 10 == 0:
#             print(f"Processed {count}/{total_combos} | Best Obj: {best_obj:.2f}     ", end='\r')
            
#     print(f"\nGrid Search Complete. Best Obj: {best_obj:.4f}")
#     return best_params, best_W, best_obj

# # ==========================================
# # 5. Visualizations (Whole Space + Per Class)
# # ==========================================
# def run_visualizations(W, alpha, l1, l2, tau, data, y_cls, y_clst, hist, hist_W, dims, best_obj, title_prefix):
#     # Project Data
#     proj = data @ W
#     u_parents = np.unique(y_cls)
    
#     # Define Semantic Names
#     names = {1: 'Footwear', 2: 'Tops/Outer', 3: 'Legs/Body'}
    
#     fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
#     # Overall Scatter
#     cmap_parents = cm.get_cmap('viridis', len(u_parents))
#     for i, c in enumerate(u_parents):
#         idx = np.where(y_cls == c)[0]
#         label_name = names.get(c, f"Parent {c}")
#         axs[0].scatter(proj[idx, 0], proj[idx, 1], label=label_name, 
#                        color=cmap_parents(i), alpha=0.7, s=20)
        
#     axs[0].set_title(f"{title_prefix} Projection (Fashion-MNIST)\n(a={alpha:.2f}, t={tau:.2f}, l1={l1:.2f}, l2={l2:.2f})")
#     axs[0].set_xlabel("Component 1")
#     axs[0].set_ylabel("Component 2")
#     axs[0].legend()
#     axs[0].grid(True, alpha=0.2)
    
#     # Gradient History
#     if hist_W and len(hist_W) > 0:
#         axs[1].plot(hist_W, label='Grad W Norm', color='purple')
#         axs[1].set_yscale('log')
#         axs[1].set_xscale('log')
#         axs[1].set_title("Gradient Norm History")
#         axs[1].set_xlabel("Iteration")
#         axs[1].legend()
#     else:
#         axs[1].text(0.5, 0.5, "No History for Grid Search Snapshot", ha='center')

#     plt.tight_layout()
#     save_path = f'figs/hlda_v4_fashion/{dims}'
#     os.makedirs(save_path, exist_ok=True)
#     plt.savefig(os.path.join(save_path, f"{title_prefix}_projection_{best_obj:.4f}.png"))
#     plt.show()

# # ==========================================
# # Main Execution
# # ==========================================
# if __name__ == "__main__":
#     # Settings
#     reg = 1e-4
    
#     # 1. Load Data
#     # NOTE: Set max_samples=None to use full 60k dataset (slower)
#     data, y_cls, y_clst = get_fashion_mnist_hierarchy(max_samples=3000)
#     dims = data.shape[1] # 784 for MNIST
    
#     print(f"2. Computing Scatter Matrices for {dims} dims...")
#     Sb, Sws, Sbs, pm, sm, mt = compute_matrices(data, y_cls, y_clst, dims)
    
#     # --- PHASE A: GRID SEARCH ---
#     best_p, best_W_grid, best_obj = run_grid_search(Sb, Sws, Sbs, sm, pm, mt, reg)
    
#     # Visualize Grid Result
#     print("\nVisualizing Best Grid Search Result...")
#     run_visualizations(best_W_grid, best_p['alpha'], best_p['l1'], best_p['l2'], best_p['tau'],
#                     data, y_cls, y_clst, [], [], dims, best_obj, title_prefix="Grid Search Best")
    
#     # --- PHASE B: GRADIENT ASCENT ---
#     print("3. Running Full Gradient Ascent using Best Grid Params...")
#     W, a, l1, l2, t, h, hW = joint_gradient_ascent_v4(
#             Sb, Sws, Sbs, sm, pm, mt, 
#             reg=reg, num_iters=10000, fix_hyperparams=best_p,
#             verbose=True
#         )
    
#     print(f"\nFinal Result: Obj={h[-1]:.4f}")
#     run_visualizations(W, a, l1, l2, t, data, y_cls, y_clst, h, hW, dims, best_obj, title_prefix="Final Gradient Ascent")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import itertools
import time
import os

# ==========================================
# 1. Real World Data Loading (Fashion MNIST)
# ==========================================
def get_fashion_mnist_hierarchy(max_samples=3000):
    """
    Fetches Fashion-MNIST and structures it into a 2-level hierarchy.
    Native Dim: 784 (28x28 images)
    """
    print("Fetching Fashion-MNIST from OpenML (this may take a moment)...")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, cache=True)
    y = y.astype(int)
    
    # Hierarchy Mapping
    # 1: Footwear (Sandal, Sneaker, Ankle Boot)
    # 2: Tops/Outer (T-shirt, Pullover, Coat, Shirt)
    # 3: Legs/Body (Trouser, Dress)
    hierarchy_map = {
        1: [5, 7, 9],       
        2: [0, 2, 4, 6],    
        3: [1, 3]           
    }
    
    # Re-indexing Logic
    subclass_reindex_map = {} 
    for pid, subclasses in hierarchy_map.items():
        for i, sub in enumerate(subclasses):
            subclass_reindex_map[sub] = (pid, i + 1)
            
    # Filter
    valid_mask = np.isin(y, [c for sublist in hierarchy_map.values() for c in sublist])
    X_filtered = X[valid_mask]
    y_filtered = y[valid_mask]
    
    # Downsample for speed
    if max_samples and max_samples < len(X_filtered):
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(X_filtered))[:max_samples]
        X_filtered = X_filtered[indices]
        y_filtered = y_filtered[indices]
    
    final_labels_class = []
    final_labels_cluster = []
    
    for label in y_filtered:
        pid, sub_idx = subclass_reindex_map[label]
        final_labels_class.append(pid)
        final_labels_cluster.append((pid, sub_idx))
        
    # Standardize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_filtered)
    # X_norm = X_filtered
    
    print(f"Data Loaded: {X_norm.shape[0]} samples, {X_norm.shape[1]} original dimensions.")
    return X_norm, np.array(final_labels_class), np.array(final_labels_cluster)

# ==========================================
# 2. Matrices & Helper Functions
# ==========================================
def compute_matrices(data, labels_class, labels_cluster, dims):
    overall_mean = np.mean(data, axis=0)
    unique_classes = np.unique(labels_class)
    S_B = np.zeros((dims, dims)); S_WS = np.zeros((dims, dims)); S_BS = np.zeros((dims, dims))
    parent_means = {}; subclass_means = {}
    mu_tot = overall_mean
    
    for c in unique_classes:
        idx = np.where(labels_class == c)[0]
        X_c = data[idx, :]; N_c = X_c.shape[0]; mu_c = np.mean(X_c, axis=0)
        parent_means[c] = mu_c
        S_B += N_c * np.outer(mu_c - overall_mean, mu_c - overall_mean)
        
        # Filter tuples for current parent class
        current_cluster_labels = labels_cluster[idx]
        unique_subs = np.unique([lbl[1] for lbl in current_cluster_labels])
        
        subclass_means[c] = []
        for sub in unique_subs:
            # Mask: Parent == c AND SubIndex == sub
            mask = np.array([(lbl[0] == c and lbl[1] == sub) for lbl in labels_cluster])
            idx_sub = np.where(mask)[0]
            
            if len(idx_sub) == 0: continue
            
            X_cs = data[idx_sub, :]
            N_cs = X_cs.shape[0]; mu_cs = np.mean(X_cs, axis=0)
            subclass_means[c].append(mu_cs)
            
            diff = X_cs - mu_cs
            S_WS += diff.T @ diff
            S_BS += N_cs * np.outer(mu_cs - mu_c, mu_cs - mu_c)
            
    return S_B, S_WS, S_BS, parent_means, subclass_means, mu_tot

def grad_norm_wrt_W(W, diff_vec, eps=1e-8):
    """Computes gradient and value of ||W^T @ diff_vec||_2"""
    proj = W.T @ diff_vec
    norm_val = np.linalg.norm(proj) + eps
    grad = np.outer(diff_vec, proj) / norm_val
    return grad, norm_val

# ==========================================
# 3. Optimization Logic (STABILIZED V4)
# ==========================================
def joint_gradient_ascent_v4(S_B, S_WS, S_BS, subclass_means, parent_means, mu_tot, 
                             reg=1e-4, num_iters=300, fix_hyperparams=None, verbose=True):
    dims = S_B.shape[0]
    # Seeding inside function for consistent starts across grid search
    np.random.seed(42) 
    W = np.random.randn(dims, 2); W, _ = np.linalg.qr(W)
    
    # Defaults
    alpha = fix_hyperparams['alpha'] if fix_hyperparams else 0.5
    l1 = fix_hyperparams['l1'] if fix_hyperparams else 0.5
    l2 = fix_hyperparams['l2'] if fix_hyperparams else 0.5
    tau = fix_hyperparams['tau'] if fix_hyperparams else 0.8  
    
    # --- STABILITY GUARD 1: Conservative Steps ---
    step_W = 1e-5      # Reduced from 1e-4
    step_hyp = 1e-4    # Reduced from 1e-3
    eps_div = 1e-3     # Stronger epsilon to prevent 1/x explosion
    
    gamma_alpha = 5.0   # Penalty for straying from 0.5
    decay_l = 0.01      # Decay for l1/l2
    
    history = []; hW = []
    
    for it in range(num_iters):
        # print(f'Iteration {it+1}/{num_iters}' , end='\r')
        # S_mix
        S_mix = alpha * S_WS + (1 - alpha) * S_BS + reg * np.eye(dims)
        
        # S_R (Numerator: Repulsion)
        S_R_val = 0.0; grad_S_R = np.zeros_like(W)
        for c, sub_means in subclass_means.items():
            num_sub = len(sub_means)
            for i in range(num_sub):
                for j in range(i+1, num_sub):
                    diff = sub_means[i] - sub_means[j]
                    # Use stable epsilon
                    grad, n_val = grad_norm_wrt_W(W, diff, eps=eps_div)
                    inv_norm = 1.0 / (n_val + eps_div)
                    S_R_val += inv_norm
                    grad_S_R += -(inv_norm**2) * grad
                    
        # S_T (Denominator: Topology)
        S_T_val = 0.0; grad_S_T = np.zeros_like(W)
        for c, sub_means in subclass_means.items():
            for mu_cs in sub_means:
                diff_anchor = mu_cs - parent_means[c]
                g_anchor, n_anchor = grad_norm_wrt_W(W, diff_anchor, eps=eps_div)
                
                diff_global = mu_cs - mu_tot
                proj_global = W.T @ diff_global
                n_global_sq = np.dot(proj_global, proj_global)
                g_global_sq = 2 * np.outer(diff_global, proj_global)
                
                loss = n_anchor - tau * n_global_sq
                
                if loss > 0:
                    S_T_val += loss
                    grad_S_T += (g_anchor - tau * g_global_sq)

        # Ratio & Penalties
        tr_num = np.trace(W.T @ S_B @ W)
        tr_den = np.trace(W.T @ S_mix @ W)
        
        N = tr_num + l2 * S_R_val
        D = tr_den + l1 * S_T_val + eps_div 
        ratio = N / D
        
        pen_alpha = gamma_alpha * (alpha - 0.5)**2
        pen_l = decay_l * (l1**2 + l2**2)
        
        # Gradients
        grad_N_W = 2 * S_B @ W + l2 * grad_S_R
        grad_D_W = 2 * S_mix @ W + l1 * grad_S_T
        grad_J_W = (D * grad_N_W - N * grad_D_W) / (D**2)
        
        # --- STABILITY GUARD 2: Gradient Clipping ---
        gnorm = np.linalg.norm(grad_J_W)
        if gnorm > 1.0:
            grad_J_W = grad_J_W / gnorm
        
        if not fix_hyperparams:
            dD_da = np.trace(W.T @ (S_WS - S_BS) @ W)
            grad_alpha = (-(N * dD_da) / (D**2)) - (2 * gamma_alpha * (alpha - 0.5))
            grad_l1 = (-(N * S_T_val) / (D**2)) - (2 * decay_l * l1)
            grad_l2 = (S_R_val / D) - (2 * decay_l * l2)
            
            l1 = max(l1 + step_hyp * grad_l1, 1e-4) 
            l2 = max(l2 + step_hyp * grad_l2, 1e-4)
            alpha = np.clip(alpha + step_hyp * grad_alpha, 0.01, 0.99)
            
        W += step_W * grad_J_W; W, _ = np.linalg.qr(W)
        history.append(ratio)
        hW.append(gnorm)
        
        if verbose and it % 50 == 0:
            print(f"Iter {it}: Ratio={ratio:.4f}, a={alpha:.2f}, l1={l1:.2f}, l2={l2:.2f} (tau={tau:.2f})", end='\r')
            
    return W, alpha, l1, l2, tau, history, hW

# ==========================================
# 4. Grid Search Wrapper
# ==========================================
def run_grid_search(S_B, S_WS, S_BS, sub_means, par_means, mu_tot, reg):
    print("Running Grid Search V4 (Ratio + Squared Hinge) on Fashion-MNIST...")
    alphas = np.linspace(0, 1, 10) # Reduced slightly for speed
    l1s = np.linspace(0, 1000, 10)
    l2s = np.linspace(0, 1000, 10)
    taus = np.linspace(0, 1, 5) 
    
    grid = {'alpha': alphas, 'l1': l1s, 'l2': l2s, 'tau': taus}
    keys, values = zip(*grid.items())
    
    best_obj = -np.inf
    best_params = None
    best_W = None
    
    total_combos = np.prod([len(v) for v in values])
    print(f"Total Combinations: {total_combos}")
    
    count = 0
    for bundle in itertools.product(*values):
        params = dict(zip(keys, bundle))
        
        # Use short iters for grid search
        W_curr, _, _, _, _, hist, _ = joint_gradient_ascent_v4(
            S_B, S_WS, S_BS, sub_means, par_means, mu_tot, 
            reg=reg, num_iters=30, fix_hyperparams=params, 
            verbose=False 
        )
        
        if hist[-1] > best_obj:
            best_obj = hist[-1]
            best_params = params
            best_W = W_curr.copy()
            
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{total_combos} | Best Obj: {best_obj:.2f}     ", end='\r')
            
    print(f"\nGrid Search Complete. Best Obj: {best_obj:.4f}")
    return best_params, best_W, best_obj

# ==========================================
# 5. Visualizations
# ==========================================
def run_visualizations(W, alpha, l1, l2, tau, data, y_cls, y_clst, hist, hist_W, dims, best_obj, title_prefix):
    # Project Data
    proj = data @ W
    u_parents = np.unique(y_cls)
    
    # Define Semantic Names
    names = {1: 'Footwear', 2: 'Tops/Outer', 3: 'Legs/Body'}
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Overall Scatter
    cmap_parents = cm.get_cmap('viridis', len(u_parents))
    for i, c in enumerate(u_parents):
        idx = np.where(y_cls == c)[0]
        label_name = names.get(c, f"Parent {c}")
        axs[0].scatter(proj[idx, 0], proj[idx, 1], label=label_name, 
                       color=cmap_parents(i), alpha=0.7, s=20)
        
    axs[0].set_title(f"{title_prefix} Projection\n(a={alpha:.2f}, t={tau:.2f}, l1={l1:.2f}, l2={l2:.2f})")
    axs[0].set_xlabel("Component 1")
    axs[0].set_ylabel("Component 2")
    axs[0].legend()
    axs[0].grid(True, alpha=0.2)
    
    # Gradient History
    if hist_W and len(hist_W) > 0:
        axs[1].plot(hist_W, label='Grad W Norm', color='purple')
        axs[1].set_yscale('log')
        # axs[1].set_xscale('log') # Can toggle linear/log x-scale
        axs[1].set_title("Gradient Norm History (Clipped)")
        axs[1].set_xlabel("Iteration")
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "No History for Grid Search Snapshot", ha='center')

    plt.tight_layout()
    save_path = f'figs/hlda_v4_stable/{dims}'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{title_prefix}_projection_{best_obj:.4f}.png"))
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    reg = 1e-4
    
    # 1. Load Data
    # NOTE: Set max_samples=None to use full 60k dataset (slower)
    data_raw, y_cls, y_clst = get_fashion_mnist_hierarchy(max_samples=10000)
    
    # --- CRITICAL STEP: PCA PREPROCESSING ---
    print("2. Applying PCA to reduce noise...")
    # Reduce 784 pixels down to 50 significant features
    # This prevents the "banana" shape and helps linear separation
    num_dims = 100
    pca = PCA(n_components=num_dims)
    data = pca.fit_transform(data_raw)
    dims = num_dims 
    
    print(f"3. Computing Scatter Matrices for {dims} dims...")
    Sb, Sws, Sbs, pm, sm, mt = compute_matrices(data, y_cls, y_clst, dims)
    
    # --- PHASE A: GRID SEARCH ---
    best_p, best_W_grid, best_obj = run_grid_search(Sb, Sws, Sbs, sm, pm, mt, reg)
    
    # Visualize Grid Result
    print("\nVisualizing Best Grid Search Result...")
    run_visualizations(best_W_grid, best_p['alpha'], best_p['l1'], best_p['l2'], best_p['tau'],
                    data, y_cls, y_clst, [], [], dims, best_obj, title_prefix="Grid Search Best")
    
    # --- PHASE B: GRADIENT ASCENT ---
    print("4. Running Full Gradient Ascent using Best Grid Params...")
    W, a, l1, l2, t, h, hW = joint_gradient_ascent_v4(
            Sb, Sws, Sbs, sm, pm, mt, 
            reg=reg, num_iters=5000, fix_hyperparams=best_p,
            verbose=True
        )
    
    print(f"\nFinal Result: Obj={h[-1]:.4f}")
    run_visualizations(W, a, l1, l2, t, data, y_cls, y_clst, h, hW, dims, best_obj, title_prefix="Final Gradient Ascent")