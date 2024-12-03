import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import numpy as np
from exercise7_kernel_SVM import *

# import
digits = datasets.load_digits()

# flatten the images
n = len(digits.images)
D = digits.images.reshape((n, -1))
y = digits.target

# Split data into 70% train and 30% test subsets
D_train, D_test, y_train, y_test = train_test_split(
    D, y, test_size=0.3, shuffle=False
)

clf = svm.SVC(gamma=0.0008, C=0.9)

# Learn the digits on the train subset
clf.fit(D_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(D_test)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, D_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

#7c
n_sv_0 = clf.n_support_[0]
n_sv_1 = clf.n_support_[1]
dual_coef = clf.dual_coef_
print(dual_coef.shape)
coefficients_0_vs_1 = dual_coef[0][:n_sv_0+n_sv_1]
num_support_vectors_0_vs_1 = (coefficients_0_vs_1 != 0).sum()
print(coefficients_0_vs_1.size)
print(num_support_vectors_0_vs_1)

#7e
values_parameters,acc=get_best_combination_and_score(D,y,[0.0001, 0.0006, 0.001, 0.006],[0.6, 0.8, 1, 2, 3, 4, 6],5)

print(
    f"7e. Beste combination values: C={values_parameters["C"]}, gamma={values_parameters["gamma"]}:\n"
    f"Accuracy: {acc}"
)

# final_model = SVC(kernel='rbf', C=values_parameters['C'], gamma=values_parameters['gamma'])
# final_model.fit(D, y)