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
from sklearn.cluster import SpectralClustering

from hfps_clustering import density_based
from classifier import Classifier
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


def breakdown_labelset(classifier, it, label_clusters):
    prediction = set(classifier.label_based.keys())
    new_classifiers = []
    label_subsets = [prediction.intersection(cluster) for cluster in label_clusters if
                     prediction.intersection(cluster).__len__() > 0]

    if label_subsets.__len__() > 1:
        classifier.update_numerosity(-1)
        for cluster in label_subsets:
            new_classifier = Classifier()
            new_classifier.classifier_copy(classifier, it)
            new_classifier.prediction = cluster
            new_classifier.label_based = {k: 0 for k in cluster}
            new_classifier.parent_prediction.append(prediction)
            new_classifier.set_fitness(INIT_FITNESS)
            new_classifier.loss = 0.0
            new_classifiers.append(new_classifier)
    if new_classifiers.__len__() > 0:
        return new_classifiers


class GraphPart:
    def __init__(self):
        self.classifiers = []
        self.label_clusters = []
        self.label_similarity = None
        self.predicted_labels = []
        self.label_matrix = []

    def build_graph(self, matching_classifiers, cosine_matrix=None):
        self.label_clusters = []
        self.label_matrix = []
        self.classifiers = matching_classifiers
        if any([classifier.prediction.__len__() > 1 for classifier in self.classifiers]):
            self.predicted_labels = sorted(list(set().union(*[classifier.prediction for classifier in self.classifiers])))

            def label_vector(classifier):
                return [max(classifier.label_based[label] / classifier.match_count, INIT_FITNESS)
                        if label in classifier.prediction else 0 for label in self.predicted_labels]

            if cosine_matrix.any():
                self.label_similarity = cosine_matrix[:, self.predicted_labels][self.predicted_labels, :]
            else:
                for classifier in matching_classifiers:
                    if classifier.match_count > 0:
                        for idx in range(classifier.numerosity):
                            self.label_matrix.append(label_vector(classifier))
                self.label_similarity = calculate_similarity(self.label_matrix)
            return True
        else:
            return False

    def refine_prediction(self, it, vote=None):
        self.label_clusters = []
        try:
            self.label_similarity = np.where(self.label_similarity > 0.1, self.label_similarity, 0)
        except TypeError:
            return [], None
        n_connected, label_connected = connected_components(self.label_similarity)
        if n_connected > 1:
            for c in range(n_connected):
                self.label_clusters.append(set([self.predicted_labels[node] for node in
                                                range(self.predicted_labels.__len__()) if label_connected[node] == c]))
        else:
            # return [], 0
            _, self.label_clusters = density_based(K, self.label_matrix, 1 - self.label_similarity,
                                                   self.predicted_labels)

            vertex_weights = np.zeros((self.predicted_labels.__len__(), self.predicted_labels.__len__()))
            i = 0
            for l in self.predicted_labels:
                vertex_weights[i][i] = max(vote[l], 0.01)
                i += 1
            sc = SpectralClustering(n_clusters=K, affinity='precomputed', n_init=10,
                                    assign_labels='discretize', weights=vertex_weights)
            sc.fit_predict(self.label_similarity)
            label_clusters = []
            for n in range(K):
                label_clusters.append(set([self.predicted_labels[idx] for idx in range(self.predicted_labels.__len__())
                                           if sc.labels_[idx] == n]))
        micro_pop_reduce = 0
        new_classifiers = []
        for classifier in self.classifiers:
            if classifier.prediction.__len__() > L_MIN:
                try:
                    [new_classifiers.append(cl) for cl in breakdown_labelset(classifier, it, self.label_clusters)]
                    micro_pop_reduce += 1
                except TypeError:
                    pass
        return new_classifiers, micro_pop_reduce
