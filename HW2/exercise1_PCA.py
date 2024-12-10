import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def PCA(D, r):
    # First compute sample mean
    mu = np.mean(D, axis=0)

    # Subtract the mean from the data
    centered_data = D - mu

    # Compute Truncated SVD
    U, S, Vt = linalg.svd(centered_data, full_matrices=False)

    # Calculate low-dimensional data and principal components
    low_dimensional_data = centered_data@Vt[:r].T

    # return low-dimensional view on the data
    return low_dimensional_data, Vt[:r]


def plot_principal_components(V_r, image_shape, num_components=10):
    plt.figure(figsize=(12, 6))
    for i in range(num_components):
        # Get the i-th principal component
        pc = V_r[i, :]  # Rows of V_r are principal components

        # Reshape it to image dimensions
        pc_image = pc.reshape(image_shape)

        # Normalize the pixel values for visualization (0–255)
        pc_image = (pc_image - np.min(pc_image)) / \
            (np.max(pc_image) - np.min(pc_image)) * 255

        # Plot the principal component
        plt.subplot(1, num_components, i + 1)
        plt.imshow(pc_image, cmap="gray")
        plt.title(f"PC {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

