import numpy as np

#5a
def proportional_probability(target, categories):
    results = {}
    counts = np.bincount(target)
    for i, category in enumerate(categories):
        results[category] = counts[i] / len(target)
    return results

#5b
def log_laplace_proabability_word_given_target(word,category,matrix,dataset,vectorizer,alpha):
    return np.log(laplace_proabability_word_given_target(word,category,matrix,dataset,vectorizer,alpha))
def laplace_proabability_word_given_target(word,category,matrix,dataset,vectorizer,alpha):
    column_index=np.where(vectorizer.get_feature_names_out() == word)[0][0]
    #we filter by category and we only focus on the given word column
    dense_matrix = matrix.toarray()  
    filtered_data = [row[column_index] for row, target in zip(dense_matrix, dataset.target) if target == category]

    number_of_repetitions_category=len(filtered_data)

    if(number_of_repetitions_category==0):
        print("Error, target no present in data")
        return 0.
    
    word_repeticion=sum(filtered_data)

    # number_possible_values= np.unique(dataset[:, column_index].toarray().flatten()).size#should be 2 since it is binary
    return (word_repeticion+alpha)/(number_of_repetitions_category+alpha*2)

