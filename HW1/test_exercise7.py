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


# 7a
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
    plt.close()

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)


# 7c
n_sv_0 = clf.n_support_[0]
n_sv_1 = clf.n_support_[1]

# we get the first n_sv_0+n_sv_1 coefficients of row 1
dual_coef = clf.dual_coef_
coefficients_0_vs_1 = dual_coef[0][:n_sv_0+n_sv_1]

# check how many of them are different from 0
num_support_vectors_0_vs_1 = (coefficients_0_vs_1 != 0).sum()
print("Number of support vectors between 0 and 1: ", num_support_vectors_0_vs_1, "\n")


# 7d
fig, axes = plt.subplots(2, 4, figsize=(15, 6))
axes = axes.ravel()

coefficients_in_0_against_1= dual_coef[0][:n_sv_0]
four_highest_0_against_1 = sorted(enumerate(coefficients_in_0_against_1), key=lambda x: abs(x[1]), reverse=True)[:4]

coffeicients_in_1_against_0 = dual_coef[0][n_sv_0:n_sv_0+n_sv_1]
four_highest_1_against_0= sorted(enumerate(coffeicients_in_1_against_0), key=lambda x:abs(x[1]), reverse=True)[:4]

for idx, (index, coef) in enumerate(four_highest_0_against_1):
    array = clf.support_vectors_[index].copy()
    image = array.reshape(8, 8)
    axes[idx].set_axis_off()
    axes[idx].imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    axes[idx].set_title(f"Class 0, α = {abs(coef):.2f}")

for idx, (index, coef) in enumerate(four_highest_1_against_0):
    array = clf.support_vectors_[n_sv_0 + index].copy()
    image = array.reshape(8, 8)
    axes[idx + 4].set_axis_off()
    axes[idx + 4].imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    axes[idx + 4].set_title(f"Class 1, α = {abs(coef):.2f}")

plt.tight_layout()
plt.show()


# 7e
values_parameters,acc=get_best_combination_and_score(D,y,[0.0001, 0.0006, 0.001, 0.006],[0.6, 0.8, 1, 2, 3, 4, 6],5)

print(
    f"7e. Best combination values: C={values_parameters["C"]}, gamma={values_parameters["gamma"]}:\n"
    f"Accuracy: {acc}"
)

# final_model = SVC(kernel='rbf', C=values_parameters['C'], gamma=values_parameters['gamma'])
# final_model.fit(D, y)