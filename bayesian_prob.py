# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# The code is adopted from the original MLkNN code under
# http://scikit.ml/_modules/skmultilearn/adapt/mlknn.html#MLkNN
# ------------------------------------------------------------------------------

from skmultilearn.base.base import MLClassifierBase
from skmultilearn.utils import get_matrix_in_format
import numpy as np
import scipy.sparse as sparse
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt


def similarity(classifier, x):
    center = [(att[1] + att[0]) / 2 for att in classifier.condition]
    x0 = [x[idx] for idx in classifier.specified_atts]
    try:
        return cosine_similarity([center, x0])[0][1]
    except ValueError:
        return 0.0


def distance(classifier, x):
    center = [(att[1] + att[0]) / 2 for att in classifier.condition]
    d = sqrt(sum([(x[att] - center[idx])**2 for (idx, att)
             in enumerate(classifier.specified_atts)]))
    return d / classifier.specified_atts.__len__()


def match(classifier, state, dtypes):
    for idx, ref in enumerate(classifier.specified_atts):
        x0 = state[ref]
        if dtypes[ref]:
            if classifier.condition[idx][0] <= x0 <= classifier.condition[idx][1]:
                pass
            else:
                return False
        else:
            if x0 == classifier.condition[idx]:
                pass
            else:
                return False
    return True


class KnnPosterior:

    def __init__(self, pop, dtypes, k=10, s=1.0, ignore_first_neighbours=0):
        """Initializes the classifier

        Parameters
        ----------
        k : int
            number of neighbours of each input instance to take into account
        s: float (default is 1.0)
            the smoothing parameter
        ignore_first_neighbours : int (default is 0)
                ability to ignore first N neighbours, useful for comparing
                with other classification software.
        pop : array of classifier objects
              trained rule base

        Attributes
        ----------
        knn_ : an instance of sklearn.NearestNeighbors
            the nearest neighbors single-label classifier used underneath
        """
        super(KnnPosterior, self).__init__()
        self.k = k  # Number of neighbours
        self.s = s  # Smooth parameter
        self.ignore_first_neighbours = ignore_first_neighbours
        self.copyable_attrs = ['k', 's', 'ignore_first_neighbours']
        self.pop = pop
        self.dtypes = dtypes

    def _compute_prior(self, y):
        """Helper function to compute for the prior probabilities

        Parameters
        ----------
        y : numpy.ndarray or scipy.sparse
            the training labels

        Returns
        -------
        numpy.ndarray
            the prior probability given true
        numpy.ndarray
            the prior probability given false
        """
        prior_prob_true = np.array((self.s + y.sum(axis=0)) / (self.s * 2 + self._num_instances))[0]
        prior_prob_false = 1 - prior_prob_true

        return prior_prob_true, prior_prob_false

    def _compute_cond(self, X, y, y_hat):
        """Helper function to compute for the posterior probabilities

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        numpy.ndarray
            the posterior probability given true
        numpy.ndarray
            the posterior probability given false
        """

        c = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='i8')
        cn = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='i8')

        # y_hat is the learned label matrix
        label_info = get_matrix_in_format(y, 'dok')
        label_hat_info = get_matrix_in_format(y_hat, 'dok')

        # contains indices of the neighbor classifiers per instance
        neighbors = [self.nearest_neighbor(x) for x in X]

        for instance in range(self._num_instances):
            deltas = label_hat_info[neighbors[instance], :].sum(axis=0)
            for label in range(self._num_labels):
                if label_info[instance, label] == 1:
                    c[label, deltas[0, label]] += 1
                else:
                    cn[label, deltas[0, label]] += 1

        c_sum = c.sum(axis=1)
        cn_sum = cn.sum(axis=1)

        cond_prob_true = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='float')
        cond_prob_false = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='float')
        for label in range(self._num_labels):
            for neighbor in range(self.k + 1):
                cond_prob_true[label, neighbor] = (self.s + c[label, neighbor]) / (
                        self.s * (self.k + 1) + c_sum[label, 0])
                cond_prob_false[label, neighbor] = (self.s + cn[label, neighbor]) / (
                        self.s * (self.k + 1) + cn_sum[label, 0])
        return cond_prob_true, cond_prob_false

    def nearest_neighbor(self, x):
        matchset = [ind for (ind, classifier) in enumerate(self.pop) if
                    match(classifier, x, self.dtypes)]
        d = [distance(self.pop[idx], x) for idx in matchset]
        d_sorted_index = sorted(range(d.__len__()), key=lambda x: d[x])
        knn_matchset = [matchset[idx] for idx in d_sorted_index[:self.k]]
        return sorted(knn_matchset)

    def fit(self, X, y):
        """Fit classifier with training data

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments.

        Returns
        -------
        self
            fitted instance of self
        """
        self._label_cache = get_matrix_in_format(y, 'lil')
        self._num_instances = self._label_cache.shape[0]
        self._num_labels = self._label_cache.shape[1]
        y_hat = []
        for cl in self.pop:
            y_hat.append([1 if label in cl.prediction else 0 for label in range(self._num_labels)])
        self._label_hat_cache = get_matrix_in_format(np.array(y_hat), 'lil')
        # Computing the prior probabilities
        self._prior_prob_true, self._prior_prob_false = self._compute_prior(self._label_cache)
        # Computing the posterior probabilities
        self._cond_prob_true, self._cond_prob_false = self._compute_cond(X, self._label_cache, self._label_hat_cache)
        return self

    def predict(self, X_test):
        """Predict labels for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse matrix of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """

        # result = sparse.lil_matrix((X_test.shape[0], self._num_labels), dtype='i8')
        result = np.zeros(shape=(self._num_labels), dtype='float')
        neighbors = self.nearest_neighbor(X_test)
        # for instance in range(X_test.shape[0]):
        deltas = self._label_hat_cache[neighbors, ].sum(axis=0)

        for label in range(self._num_labels):
            p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[0, label]]
            p_false = self._prior_prob_false[label] * self._cond_prob_false[label, deltas[0, label]]
            result[label] = int(p_true >= p_false)
        return result

    def predict_prob(self, X_test):
        """Predict probabilities of label assignments for X

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse matrix of int
            binary indicator matrix with label assignment probabilities
            with shape :code:`(n_samples, n_labels)`
        """

        result = np.zeros(shape=(self._num_labels), dtype='float')
        neighbors = self.nearest_neighbor(X_test)
        # for instance in range(X_test.shape[0]):
        deltas = self._label_hat_cache[neighbors,].sum(axis=0)

        for label in range(self._num_labels):
            p_true = self._prior_prob_true[label] * self._cond_prob_true[label, deltas[0, label]]
            result[label] = p_true
        return result
