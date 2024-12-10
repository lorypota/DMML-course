from exercise1_PCA import PCA, plot_principal_components
from sklearn.datasets import fetch_olivetti_faces
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
