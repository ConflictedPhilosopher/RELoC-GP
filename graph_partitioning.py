# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components

from hfps_clustering import density_based
from classifier import Classifier
from visualization import plot_graph
from config import *


def calculate_similarity(label_matrix, measure=0):
    if measure == 0:  # cosine similarity
        label_matrix_sparse = sparse.csr_matrix(np.array(label_matrix).transpose())
        return cosine_similarity(label_matrix_sparse)

    elif measure == 1:  # Hamming distance-based similarity
        similarity = np.zeros([NO_LABELS, NO_LABELS])
        for i in range(NO_LABELS):
            first_label = [label[i] for label in label_matrix]
            for j in range(i + 1, NO_LABELS):
                second_label = [label[j] for label in label_matrix]
                similarity[i, j] = np.sum(
                    np.array([1 for (l1, l2) in zip(first_label, second_label) if l1 == l2])) \
                    / len(label_matrix)
                similarity[j, i] = similarity[i, j]
        return similarity
    else:  # co-occurrence-based similarity
        similarity = np.zeros([NO_LABELS, NO_LABELS])
        for i in range(NO_LABELS):
            for j in range(NO_LABELS):
                first_label = [label[i] for label in label_matrix]
                second_label = [label[j] for label in label_matrix]
                similarity[i, j] = np.dot(first_label, second_label) / np.linalg.norm(second_label, 1)


class GraphPart:
    def __init__(self):
        self.classifiers = []
        self.label_clusters = []

    def refine_prediction(self, matching_classifiers, it, label_ref):
        self.label_clusters = []
        label_matrix = []
        self.classifiers = matching_classifiers
        if any([classifier.prediction.__len__() > 1 for classifier in self.classifiers]):
            predicted_labels = sorted(list(set().union(*[classifier.prediction for classifier in self.classifiers])))

            def label_vector(prediction):
                return [1 if label in prediction else 0 for label in predicted_labels]

            for classifier in matching_classifiers:
                for idx in range(classifier.numerosity):
                    label_matrix.append(label_vector(classifier.prediction))
            label_similarity = calculate_similarity(label_matrix)
            n_connected, label_connected = connected_components(label_similarity)

            if n_connected > 1:
                for c in range(n_connected):
                    temp = [predicted_labels[node] for node in range(predicted_labels.__len__())
                                if label_connected[node] == c]
                    self.label_clusters.append(set(temp))
            else:
                self.label_clusters = density_based(K, label_matrix, 1 - label_similarity, predicted_labels)
            cluster_dict = {k:self.label_clusters[k] for k in range(self.label_clusters.__len__())}
            plot_graph(cluster_dict, np.array(label_matrix), label_similarity, label_ref)
            new_classifiers = [self.breakdown_labelset(classifier, it) for classifier in self.classifiers if
                               classifier.prediction.__len__() > L_MIN]
            return new_classifiers
        else:
            return

    def breakdown_labelset(self, classifier, it):
        prediction = classifier.prediction
        new_classifiers = []
        label_subsets = [prediction.intersection(cluster) for cluster in self.label_clusters if
                         prediction.intersection(cluster).__len__() > 0]

        if label_subsets.__len__() > 1:
            classifier.update_numerosity(-1)
            for cluster in label_subsets:
                new_classifier = Classifier()
                new_classifier.classifier_copy(classifier, it)
                new_classifier.prediction = cluster
                new_classifier.parent_prediction.append(prediction)
                new_classifier.set_fitness(INIT_FITNESS)
                new_classifiers.append(new_classifier)
        return new_classifiers
