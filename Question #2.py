#Imports 
#1 - 2
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#3
from sklearn.decomposition import KernelPCA

#4 - 5
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#1 Generate Swiss roll dataset. [5 points]

    # Generate Swiss roll dataset
X, t = make_swiss_roll(n_samples=1000, noise=0.1)

#2 Plot the resulting generated Swiss roll dataset. [2 points]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.rainbow)
ax.set_title("Swiss Roll Dataset Gabriel Velazquez Berrueta Question #2")
plt.show()

#3 Use Kernel PCA (kPCA) with linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points). [6 points]

#apply Kernel PCA and return the transformed data
def apply_kpca(X, kernel, gamma=None):
    kpca = KernelPCA(n_components=2, kernel=kernel, gamma=gamma)
    X_kpca = kpca.fit_transform(X)
    return X_kpca

# Apply Kernel PCA with linear, RBF, and sigmoid kernels
X_kpca_linear = apply_kpca(X, kernel="linear")
X_kpca_rbf = apply_kpca(X, kernel="rbf", gamma=0.04)
X_kpca_sigmoid = apply_kpca(X, kernel="sigmoid", gamma=0.01)

#4 Plot the kPCA results of applying the linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points) from (3). Explain and compare the results [6 points]

# plot the kPCA results
def plot_kpca(X_kpca, t, kernel_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=t, cmap=plt.cm.Spectral)
    plt.title(f"kPCA with {kernel_name} kernel")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show()

# Plot the results for each kernel (asKED LINEAR, RBF & SIGMOID)
plot_kpca(X_kpca_linear, t, "linear")
plot_kpca(X_kpca_rbf, t, "RBF")
plot_kpca(X_kpca_sigmoid, t, "sigmoid")

#5 Using kPCA and a kernel of your choice, apply Logistic Regression for classification. Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best classification accuracy at the end of the pipeline. Print out best parameters found by GridSearchCV. [14 points]

# Define the pipeline: StandardScaler, kPCA, and Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kpca', KernelPCA(n_components=2)),
    ('log_reg', LogisticRegression(solver='lbfgs', max_iter=1000))
])

# Define parameter grid for GridSearchCV
param_grid = [
    {
        'kpca__kernel': ['linear', 'rbf', 'sigmoid'],
        'kpca__gamma': [0.01, 0.04, 0.1],  # Only for RBF and sigmoid
        'log_reg__C': [0.1, 1, 10, 100]   # Regularization parameter
    }
]

# Perform GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1)
grid_search.fit(X, t > t.mean())  # Use binary classification problem (t > mean as threshold)

# Print the best parameters found by GridSearchCV
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)


#6 Plot the Results from GridSearchCV [2 points]

# Extract the best Kernel PCA projection
best_kpca = grid_search.best_estimator_.named_steps['kpca']
X_best_kpca = best_kpca.transform(X)

# Plot the best kPCA result
plot_kpca(X_best_kpca, t, "BEST OF THE BEST kernel (from GridSearchCV)")
