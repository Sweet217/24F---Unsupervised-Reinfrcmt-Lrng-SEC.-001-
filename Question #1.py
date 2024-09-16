#1 . Retrieve and load the mnist_784 dataset of 70,000 instances. [5 points]
#pip install numpy matplotlib scikit-learn


#Imports
from sklearn.datasets import fetch_openml #2
import matplotlib.pyplot as plt #2
from sklearn.decomposition import PCA #3
from sklearn.decomposition import IncrementalPCA #5
#1 Load MNIST_748 dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

#2 Display each digit. [5 points]

def display_digits(X, y, num_digits=10):
    fig, axes = plt.subplots(1, num_digits, figsize=(10, 5))
    for i in range(num_digits):
        axes[i].imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Digit: {y[i]}')
        axes[i].axis('off')
    plt.show()

display_digits(X, y)

#3 Use PCA to retrieve the 1st and 2nd principal component and output their explained variance ratio. [5 points]

# Initialize PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Explained variance ratio
print(f'Explained variance ratio: {pca.explained_variance_ratio_}')

#4 Plot the projections of the 1st and 2nd principal component onto a 2D hyperplane. [5 points]

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='tab10', alpha=0.5)
plt.colorbar()
plt.title('PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

#5 Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions. [10 points]

ipca = IncrementalPCA(n_components=154)
X_ipca = ipca.fit_transform(X)


#6 Display the original and compressed digits from (5). [5 points]

X_reconstructed = ipca.inverse_transform(X_ipca)

def display_original_and_compressed(X, X_reconstructed, y, num_digits=10):
    fig, axes = plt.subplots(2, num_digits, figsize=(15, 6))
    
    for i in range(num_digits):
        # OG IMAGE
        original_image = X.iloc[i].values.reshape(28, 28)
        axes[0, i].imshow(original_image, cmap='gray')
        axes[0, i].set_title(f'Original: {y[i]}')
        axes[0, i].axis('off')
        #R/C IMAGE
        reconstructed_image = X_reconstructed[i].reshape(28, 28)
        axes[1, i].imshow(reconstructed_image, cmap='gray')
        axes[1, i].set_title(f'Compressed: {y[i]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

display_original_and_compressed(X, X_reconstructed, y)
