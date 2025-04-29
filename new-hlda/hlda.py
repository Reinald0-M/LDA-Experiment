# Export the full Python implementation code as a downloadable .py file

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from collections import defaultdict

# Data generation
def generate_data(sigma=0.5):
    np.random.seed(0)
    d = 6
    clusters_per_class = {1: [120], 2: [40, 40, 40], 3: [30, 30, 30, 30]}
    base_means = {1: np.zeros(d), 2: np.array([5, 0] + [0]*(d-2)), 3: np.array([0, 5] + [0]*(d-2))}
    sub_means = {
        1: [base_means[1]],
        2: [base_means[2] + np.array([0, i, 0, 0, 0, 0]) for i in [-1.5, 0, 1.5]],
        3: [base_means[3] + np.array([i, 0, 0, 0, 0, 0]) for i in [-2, -1, 1, 2]]
    }

    X_list, y_class, y_sub = [], [], []
    for cls, sizes in clusters_per_class.items():
        for sub_idx, size in enumerate(sizes):
            points = np.random.randn(size, d) * sigma + sub_means[cls][sub_idx]
            X_list.append(points)
            y_class += [cls]*size
            y_sub += [sub_idx+1]*size
    X = np.vstack(X_list)
    y_class, y_sub = np.array(y_class), np.array(y_sub)
    return X, y_class, y_sub

# Standardize
def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

# Scatter matrices
def scatter_matrices(X, y_class, y_sub):
    d = X.shape[1]
    classes = np.unique(y_class)
    m_global = X.mean(axis=0)
    S_b = np.zeros((d, d))
    S_w = np.zeros((d, d))
    S_ws = np.zeros((d, d))
    S_bs = np.zeros((d, d))
    subclass_means = {}
    for c in classes:
        Xc = X[y_class == c]
        m_c = Xc.mean(axis=0)
        S_b += Xc.shape[0] * np.outer(m_c - m_global, m_c - m_global)
        for s in np.unique(y_sub[y_class == c]):
            Xs = X[(y_class == c) & (y_sub == s)]
            m_s = Xs.mean(axis=0)
            subclass_means[(c, s)] = m_s
            S_ws += (Xs - m_s).T @ (Xs - m_s)
            S_bs += Xs.shape[0] * np.outer(m_s - m_c, m_s - m_c)
            S_w += (Xs - m_c).T @ (Xs - m_c)
    return S_b, S_w, S_ws, S_bs, subclass_means

# Projection matrix
def lda_projection(S_b, S_w):
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_w) @ S_b)
    idx = np.argsort(eigvals)[::-1][:2]
    return eigvecs[:, idx]

# k-NN
def knn_classify(X_train, y_train, X_test, k=5):
    preds = []
    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)
        nn = y_train[np.argsort(dists)[:k]]
        vals, counts = np.unique(nn, return_counts=True)
        preds.append(vals[np.argmax(counts)])
    return np.array(preds)

# Mahalanobis classifier
def mahalanobis_classify(X_train, y_train, X_test):
    classes = np.unique(y_train)
    means = {c: X_train[y_train == c].mean(axis=0) for c in classes}
    cov = np.cov(X_train.T)
    inv_cov = np.linalg.inv(cov)
    preds = []
    for x in X_test:
        dists = [distance.mahalanobis(x, means[c], inv_cov) for c in classes]
        preds.append(classes[np.argmin(dists)])
    return np.array(preds)

# Accuracy
def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

# Plotting
def plot_proj(X_proj, y_class, y_sub, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    markers = ['o', 's', '^', 'D', 'P', '*', 'v']
    default_colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta']
    color_map = {int(k): v for k, v in zip(np.unique(y_class), default_colors)}

    for c in np.unique(y_class):
        c = int(c)
        subs = np.unique(y_sub[y_class == c])
        for i, s in enumerate(subs):
            mask = (y_class == c) & (y_sub == s)
            marker = markers[i % len(markers)]
            color = color_map.get(c, 'gray')
            # Ensure all components are passed as native Python types
            plt.scatter(
                X_proj[mask, 0],
                X_proj[mask, 1],
                c=[color], marker=marker,
                label=f'{int(c)}.{int(s)}',
                edgecolor='k', alpha=0.8
            )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.show()



# Full pipeline
def run_method(name, X, y_class, y_sub, projection_fn):
    # Train/test split
    np.random.seed(42)
    idx = np.arange(len(X))
    train_idx, test_idx = [], []
    for c in np.unique(y_class):
        for s in np.unique(y_sub[y_class == c]):
            i = idx[(y_class == c) & (y_sub == s)]
            np.random.shuffle(i)
            split = int(0.7 * len(i))
            train_idx += list(i[:split])
            test_idx += list(i[split:])
    X_train, X_test = X[train_idx], X[test_idx]
    y_class_train, y_class_test = y_class[train_idx], y_class[test_idx]
    y_sub_train, y_sub_test = y_sub[train_idx], y_sub[test_idx]

    # Scatter matrices
    S_b, S_w, S_ws, S_bs, _ = scatter_matrices(X_train, y_class_train, y_sub_train)

    # Projection matrix
    W = projection_fn(S_b, S_w, S_ws, S_bs)
    X_train_proj = X_train @ W
    X_test_proj = X_test @ W

    # Classification
    knn_acc = accuracy(y_class_test, knn_classify(X_train_proj, y_class_train, X_test_proj))
    mah_acc = accuracy(y_class_test, mahalanobis_classify(X_train_proj, y_class_train, X_test_proj))

    # Visualization
    plot_proj(X_train_proj, y_class_train, y_sub_train, f"{name} - Train")
    plot_proj(X_test_proj, y_class_test, y_sub_test, f"{name} - Test")
    print(f"{name} - 5NN Accuracy: {knn_acc*100:.2f}%, Mahalanobis Accuracy: {mah_acc*100:.2f}%\\n")

# Run all methods
X, y_class, y_sub = generate_data(sigma=0.5)
X = standardize(X)

run_method("Naive hLDA", X, y_class, y_sub,
           lambda Sb, Sw, S_ws, S_bs: lda_projection(Sb, 0.5*S_ws + 0.5*S_bs))

run_method("Regularized hLDA", X, y_class, y_sub,
           lambda Sb, Sw, S_ws, S_bs: lda_projection(Sb + S_bs, Sw + S_bs))

run_method("Multi-level hLDA", X, y_class, y_sub,
           lambda Sb, Sw, S_ws, S_bs: lda_projection(Sb + S_bs, Sw + S_ws))

