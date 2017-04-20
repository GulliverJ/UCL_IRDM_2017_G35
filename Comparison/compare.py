import pickle
from math import sqrt


def similarity(bow1, bow2):
    """
    Compares file similarity on two bag of words representations
    :param bow1: first bag of words
    :param bow2: second bag of words
    :return: similarity between 0 and 1
    """
    dot_prod, file1_norm, file2_norm = 0.0, 0.0, 0.0
    for word, count1 in bow1.items():
        count2 = bow2[word]
        dot_prod += count1 * count2
        file1_norm += count1 ** 2
        file2_norm += count2 ** 2
    norm = sqrt(file1_norm * file2_norm)
    if norm > 0:
        return dot_prod / norm
    return 0


def intersection(ranking, true_set):
    """
    Find out which documents in the ranking correspond with documents in the true_set.
    :param ranking: a list of PIDs
    :param truth: a list of bag_of_word representations of documents in the true_set.
    :return: vector of 1 or 0 values.
             if ranking[i] is in truth then return_vec[i] = 1
             otherwise return_vec[i] = 0
    """
    return_vec = []

    for item in ranking:
        file_result = listing_conversion["%d.json" % item]
        in_true_set = 0

        for item in true_set.values():
            if similarity(file_result["bag_of_words"], item) > 0.9:
                in_true_set = 1
                break

        return_vec.append(in_true_set)

    return return_vec


if __name__ == "__main__":

    listing_conversion = pickle.load(open("filenames2results.pkl", "rb"))
    true_set = pickle.load(open("Google_Pickles/alan_turing.pkl", "rb"))
    ranking = [67, 66, 5385, 5376, 5086, 5073, 4541, 3853, 3, 27324]

    vec = intersection(ranking, true_set)
    print(vec)
