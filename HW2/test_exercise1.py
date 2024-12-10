import matplotlib.pyplot as plt
from exercise1_PCA import PCA, plot_principal_components
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

faces = fetch_olivetti_faces()
D = faces.data

# 1a
# Compute and plot the first 10 principal components
_, V_r = PCA(D, 10)
plot_principal_components(V_r, (64, 64), num_components=10)

# 1b
low_dimensional_data, V_r = PCA(D, 4)
eighth_data_point_coordinates = low_dimensional_data[7]
print("Coordinates of the 8th data point in the low-dimensional space:",
      eighth_data_point_coordinates)

# 1c
low_dimensional_data, V_r = PCA(D, 90)
low_dim_vector = np.array([0.45] * 45 + [-0.45] * 45)

# Reconstruct the face
reconstructed_face = np.dot(low_dim_vector, V_r) + np.mean(D, axis=0)

# Reshape into 2D for visualization
reconstructed_image = reconstructed_face.reshape((64, 64))

# Plot the reconstructed face
plt.imshow(reconstructed_image, cmap='gray')
plt.title("Reconstructed Face")
plt.axis('off')
plt.show()