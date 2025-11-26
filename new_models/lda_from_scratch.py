from sklearn.datasets import load_iris
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
raw_data = data.copy()
data.insert(0, 'target', iris.target)

targets = data['target'].unique()
c1 = data[data['target'] == targets[0]].drop('target', axis=1)
c2 = data[data['target'] == targets[1]].drop('target', axis=1)
c3 = data[data['target'] == targets[2]].drop('target', axis=1)

mu_1 = c1.mean()
mu_2 = c2.mean()
mu_3 = c3.mean()

total_samples = c1.shape[0] + c2.shape[0] + c3.shape[0]
mu_tot = (mu_1 * c1.shape[0] + mu_2 * c2.shape[0] + mu_3 * c3.shape[0]) / total_samples

mu_class = pd.concat([mu_1, mu_2, mu_3], axis=1)


S_B = (mu_class - mu_tot.values.reshape(-1, 1)).dot((mu_class - mu_tot.values.reshape(-1, 1)).T)
S_W = (c1 - mu_1).T.dot(c1 - mu_1) + (c2 - mu_2).T.dot(c2 - mu_2) + (c3 - mu_3).T.dot(c3 - mu_3)


W = np.linalg.inv(S_W).dot(S_B)

eigvals, eigvecs = np.linalg.eig(W)
sorted_indices = np.argsort(eigvals)[::-1]
eigvals = eigvals[sorted_indices]
eigvecs = eigvecs[:, sorted_indices]

print("Eigenvalues:\n", eigvals)
print("\nEigenvectors:\n", eigvecs)
k  =3
#k = len(eigvals)  
W_lda = eigvecs[:, :k]


X = raw_data.values
X_lda = X.dot(W_lda)

print("Projected data:\n", X_lda)

targets = data['target']
colors = ['r', 'g', 'b']

combinations = [(0, 1), (0, 2), (1, 2)]
titles = ["LD1 vs LD2", "LD1 vs LD3", "LD2 vs LD3"]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, (x_idx, y_idx) in enumerate(combinations):
    ax = axes[i]
    for target, color in zip(targets.unique(), colors):
        indices = targets == target
        ax.scatter(X_lda[indices, x_idx], X_lda[indices, y_idx], label=f'Class {target}', color=color, alpha=0.7)
    ax.set_title(titles[i], fontsize=16)
    ax.set_xlabel(f"LD{x_idx + 1}", fontsize=14)
    ax.set_ylabel(f"LD{y_idx + 1}", fontsize=14)
    ax.legend(title="Classes", fontsize=12)

plt.tight_layout()
plt.show()

