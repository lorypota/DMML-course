import numpy as np

#5a
def proportional_probability(target, category):
    counts = np.bincount(target)
    return counts[category] / len(target)

def proportional_probabilities(target, categories):
    results = {}
    counts = np.bincount(target)
    length = len(target)
    for i, category in enumerate(categories):
        results[category] = counts[i] / length
    return results

#5b
def log_laplace_proabability_word_given_target(word, category, matrix, dataset, vectorizer, alpha):
    return np.log(laplace_proabability_word_given_target(word, category, matrix, dataset, vectorizer, alpha))


def laplace_proabability_word_given_target(word, category, matrix, dataset, vectorizer, alpha):
    column_index = np.where(vectorizer.get_feature_names_out() == word)[0][0]
    # filter by category and only focus on the given word column
    dense_matrix = matrix.toarray()
    filtered_data = [row[column_index] for row, target in zip(dense_matrix, dataset.target) if target == category]

    number_of_repetitions_category = len(filtered_data)

    if (number_of_repetitions_category == 0):
        print("Error, target no present in data")
        return 0

    word_repeticion = sum(filtered_data)

    # number_possible_values= np.unique(dataset[:, column_index].toarray().flatten()).size
    # 2 since it is binary
    return (word_repeticion+alpha)/(number_of_repetitions_category+alpha*2)


# 5c
category_names = {
    0: 'rec.autos',
    1: 'rec.motorcycles',
    2: 'rec.sport.baseball',
    3: 'rec.sport.hockey'
}

def compute_posterior_probability(word, category, matrix, dataset, vectorizer, categories):
    # P(Word|Category)
    likelihood = laplace_proabability_word_given_target(word, category, matrix, dataset, vectorizer, 0)

    # P(Category)
    probabilities = proportional_probabilities(dataset.target, categories)

    # P(Word) = sum(P(Word|Category_i) * P(Category_i))
    evidence = 0
    for c in categories:
        evidence += laplace_proabability_word_given_target(word, c, matrix, dataset, vectorizer, 0) * probabilities[c]

    print(probabilities)

    return (likelihood * probabilities[category_names[category]]) / evidence
