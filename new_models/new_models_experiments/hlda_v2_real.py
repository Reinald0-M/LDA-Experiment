import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import itertools
import time
import os

# ----------------------
# 1. Real World Data Loading (Fashion MNIST)
# ----------------------
def get_fashion_mnist_hierarchy(max_samples=5000):
    """
    Fetches Fashion-MNIST and structures it into a 2-level hierarchy.
    Native Dim: 784 (28x28 images)
    """
    print("Fetching Fashion-MNIST from OpenML (this may take a moment)...")
    # Fetch data (cache=True saves it locally for faster subsequent runs)
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, cache=True)
    
    # Convert labels to integers
    y = y.astype(int)
    
    # --- Define Hierarchy ---
    # Original Classes:
    # 0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
    # 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
    
    # Structure: Parent ID -> [List of Original Class IDs]
    hierarchy_map = {
        1: [5, 7, 9],       # Parent 1: Footwear
        2: [0, 2, 4, 6],    # Parent 2: Tops/Outerwear
        3: [1, 3]           # Parent 3: Legs/Body (Excluding Bag (8) for cleaner clusters)
    }
    
    # Filter and Re-label
    valid_indices = []
    labels_class = []   # Parent
    labels_cluster = [] # (Parent, Original_Subclass)
    
    # Create a mapping for original class ID to a cleaner 1-N index per parent
    # (e.g. Footwear has subclasses 1, 2, 3 instead of 5, 7, 9)
    subclass_reindex_map = {} 
    
    for pid, subclasses in hierarchy_map.items():
        for i, sub in enumerate(subclasses):
            subclass_reindex_map[sub] = (pid, i + 1)
            
    # Collect indices
    valid_mask = np.isin(y, [c for sublist in hierarchy_map.values() for c in sublist])
    X_filtered = X[valid_mask]
    y_filtered = y[valid_mask]
    
    # Downsample if requested (to keep optimization fast)
    if max_samples and max_samples < len(X_filtered):
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(X_filtered))[:max_samples]
        X_filtered = X_filtered[indices]
        y_filtered = y_filtered[indices]
    
    # Construct Label Arrays
    final_labels_class = []
    final_labels_cluster = []
    
    for label in y_filtered:
        pid, sub_idx = subclass_reindex_map[label]
        final_labels_class.append(pid)
        final_labels_cluster.append((pid, sub_idx))
        
    # Standardize Data (Critical for real world data)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_filtered)
    
    print(f"Data Loaded: {X_norm.shape[0]} samples, {X_norm.shape[1]} dimensions.")
    return X_norm, np.array(final_labels_class), np.array(final_labels_cluster)

# ----------------------
# 2. Matrices & Helper (Unchanged)
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
        
        # Extract subclasses for this parent
        # labels_cluster is a list of tuples, need to filter correctly
        current_cluster_labels = labels_cluster[idx]
        
        # Get unique subclass indices for this parent
        # The tuple is (Parent, SubIndex), so we look at index 1
        unique_subs = np.unique([lbl[1] for lbl in current_cluster_labels])
        
        subclass_means[c] = []
        for sub in unique_subs:
            # Mask relative to the FULL dataset
            # We need points that are Parent=c AND Sub=sub
            # Since labels_cluster is numpy array of objects (tuples), we interpret carefully:
            mask = np.array([ (l[0]==c and l[1]==sub) for l in labels_cluster ])
            
            idx_sub = np.where(mask)[0]
            X_cs = data[idx_sub, :]
            
            if X_cs.shape[0] == 0: continue
            
            N_cs = X_cs.shape[0]; mu_cs = np.mean(X_cs, axis=0)
            subclass_means[c].append(mu_cs)
            diff = X_cs - mu_cs
            S_WS += diff.T @ diff
            S_BS += N_cs * np.outer(mu_cs - mu_c, mu_cs - mu_c)
            
    return S_B, S_WS, S_BS, parent_means, subclass_means, mu_tot

# ----------------------
# 3. Optimization V2 (Ratio Trace) (Unchanged)
# ----------------------
def joint_gradient_ascent(S_B, S_WS, S_BS, subclass_means, parent_means, mu_tot, reg, num_iters=5000, fix_params=None):
    dims = S_B.shape[0]
    np.random.seed(42) # For reproducibility
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
        N_mat = W.T @ S_B @ W
        D_mat = W.T @ S_W @ W + eps * np.eye(2)
        D_inv = np.linalg.inv(D_mat)
        
        lda_val = np.trace(D_inv @ N_mat)
        
        term1 = 2 * S_B @ W @ D_inv
        K = D_inv @ N_mat @ D_inv 
        term2 = 2 * S_W @ W @ K
        grad_W = term1 - term2
        
        grad_alpha = -np.trace(W.T @ (S_WS - S_BS) @ W @ K)
        
        # --- Regularizers ---
        # R1 (Squared Separation)
        g_R1 = np.zeros_like(W)
        for c, means in subclass_means.items():
            for i in range(len(means)):
                for j in range(i+1, len(means)):
                    diff = means[i] - means[j]
                    proj = W.T @ diff
                    sq_val = np.dot(proj, proj) + eps
                    coeff = -1.0 / (sq_val**2)
                    grad_sq_norm = 2 * np.outer(diff, proj)
                    g_R1 += coeff * grad_sq_norm
                    
        # R2 (Hinge with Squared Anchor)
        g_R2 = np.zeros_like(W)
        for c, means in subclass_means.items():
            for mu in means:
                d_anc = mu - parent_means[c]; p_anc = W.T @ d_anc
                d_glo = mu - mu_tot;          p_glo = W.T @ d_glo
                n_anc_sq = np.dot(p_anc, p_anc)
                n_glo_sq = np.dot(p_glo, p_glo)
                
                if (n_anc_sq - tau * n_glo_sq) > 0:
                    g_anc_sq = 2 * np.outer(d_anc, p_anc)
                    g_glo_sq = 2 * np.outer(d_glo, p_glo)
                    g_R2 += (g_anc_sq - tau * g_glo_sq)
        
        # Total Gradient
        W += step * (grad_W + l1*g_R1 + l2*g_R2)
        W, _ = np.linalg.qr(W)
        
        if not fix_params:
            alpha = np.clip(alpha + step * grad_alpha, 0, 1)
            
        history.append(lda_val)
        hW.append(np.linalg.norm(grad_W))
        
    return W, l1, l2, alpha, tau, history, hW

# ----------------------
# 4. Grid Search (Unchanged)
# ----------------------
def run_grid(S_B, S_WS, S_BS, sm, pm, mt, reg):
    print("Running Grid Search V2 (Ratio Trace)...") 
    # Reduced grid for speed on real data
    alphas = np.linspace(0, 1, 10)
    # alphas = [0, 0.5, 1]
    l1s = np.linspace(0.1, 1000, 10)
    l2s = np.linspace(0.1, 1000, 10)
    # taus = np.linspace(0, 1, 10)
    taus = [0, 0.5, 1]
    
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
        # Fewer iters for grid search
        W_curr,_,_,_,_, hist, _, = joint_gradient_ascent(S_B, S_WS, S_BS, sm, pm, mt, reg, 50, params)
        
        if hist[-1] > best_obj: 
            best_obj = hist[-1]
            best_params = params
            best_W = W_curr.copy()
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{total_combos} | Best Obj: {best_obj:.2f}", end='\r')
    print(f"\nGrid Search Complete. Best Obj: {best_obj:.4f}")
    return best_params, best_W, best_obj

# ----------------------
# 5. Visualization (Unchanged)
# ----------------------
def viz(W, a, l1, l2, t, data, y_cls, y_clst, hist_W, dims, best_obj, title_prefix="V2"):
    d2 = data @ W
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Map classes to names for the plot
    class_names = {1: 'Footwear', 2: 'Tops/Outer', 3: 'Legs/Body'}
    
    # 1. Overall Scatter
    for c in np.unique(y_cls):
        idx = np.where(y_cls == c)
        axs[0].scatter(d2[idx, 0], d2[idx, 1], label=class_names.get(c, c), alpha=0.6)
        
    axs[0].set_title(f"{title_prefix} Projection (Fashion MNIST)\n(a={a:.2f}, l1={l1:.2f}, l2={l2:.2f}, t={t:.2f})")
    axs[0].set_xlabel("Component 1")
    axs[0].set_ylabel("Component 2")
    axs[0].legend()
    
    if hist_W and len(hist_W) > 0:
        axs[1].plot(hist_W, label='Grad W Norm', color='purple')
        axs[1].set_yscale('log')
        axs[1].set_title("Gradient Norm History")
        axs[1].set_xlabel("Iteration")
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "No History for Grid Search Snapshot", ha='center')
    
    plt.tight_layout()
    save_path = f'figs/hlda_real/{dims}'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{title_prefix}_projection_{best_obj:.4f}.png"))
    plt.show()

if __name__ == "__main__":
    # Real data has fixed dimensionality (784 for Fashion MNIST)
    # We remove the loop over `dim_list` to focus on the real data structure.
    
    reg = 1e-6
    
    # 1. Load Real Data
    # max_samples=3000 ensures the code runs quickly for testing. 
    # Set to None to use full dataset (60k+ images).
    d, yc, ycl = get_fashion_mnist_hierarchy(max_samples=3000)
    dims = d.shape[1] 
    
    print(f"2. Computing Matrices for dims={dims}...")
    Sb, Sw, Sbs, pm, sm, mt = compute_matrices(d, yc, ycl, dims)
    
    # --- PHASE A: GRID SEARCH ---
    best_p, best_W_grid, best_obj = run_grid(Sb, Sw, Sbs, sm, pm, mt, reg)
    
    # Visualize Grid Result
    print("\nVisualizing Best Grid Search Result...")
    viz(best_W_grid, best_p['alpha'], best_p['l1'], best_p['l2'], best_p['tau'], 
        d, yc, ycl, [], dims, best_obj, title_prefix="Grid Search Best")
    
    # --- PHASE B: GRADIENT ASCENT ---
    print("3. Running Full Gradient Ascent using Best Grid Params...")
    W_final, l1, l2, a, t, h, hW = joint_gradient_ascent(
        Sb, Sw, Sbs, sm, pm, mt, reg, 
        num_iters=1000, 
        fix_params=best_p
    )
    
    # Visualize Final Result
    print(f"Final Objective: {h[-1]:.4f}")
    viz(W_final, a, l1, l2, t, d, yc, ycl, hW, dims, best_obj, title_prefix="Final Gradient Ascent")