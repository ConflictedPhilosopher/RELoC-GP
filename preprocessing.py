# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
from os.path import join
from math import sqrt

import pandas as pd
from numpy import array, cov
from scipy.sparse import lil_matrix, csr_matrix
from scipy.linalg import inv
from numpy import argsort, mean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.cluster import KMeans
import numpy as np

from visualization import plot_bar, plot_heatmap, plot_graph
from config import *


def select_features(data_train, data_test):
    x_train = data_train.iloc[:, :NO_FEATURES]
    y_train = data_train.iloc[:, NO_FEATURES:-1]
    x_train_sp = lil_matrix(x_train).toarray()
    y_train_sp = lil_matrix(y_train).toarray()

    forest = ExtraTreesClassifier(n_estimators=100, random_state=SEED_NUMBER)
    classifier = BinaryRelevance(forest)
    classifier.fit(x_train_sp, y_train_sp)
    feature_scores = [forest.feature_importances_ for forest in classifier.classifiers_]

    indices = [argsort(importance)[::-1] for importance in feature_scores]
    selected_per_class = [index[:int(0.1 * NO_FEATURES)].tolist() for index in indices]
    selected_union = list(set().union(*selected_per_class))

    avg_feature_scores = mean(feature_scores, axis=0)
    avg_indices = argsort(avg_feature_scores)[::-1]
    selected_avg = avg_indices[:int(0.1 * NO_FEATURES)]
    drop_col = [idx for idx in range(NO_FEATURES) if idx not in selected_union]

    train_red = data_train.drop(data_train.columns[drop_col], axis=1)
    test_red = data_test.drop(data_test.columns[drop_col], axis=1)
    return train_red, test_red, selected_union.__len__()


class Preprocessing:
    def __init__(self):
        self.label_count = 0
        self.no_features = NO_FEATURES
        self.label_ref = dict()
        self.distinct_lp_count = 0
        self.unseen_test_lp = []
        self.unseen_test_labels = set()
        self.card = 0.0
        self.density = 0.0
        self.class_ratio = dict()
        self.imbalance_lp = 0
        self.imbalance_mean = 0.0
        self.cvir = 0.0
        self.attribute_info = []
        self.dtypes = []
        self.cov_inv = None
        self.id = False
        self.data_train_list = []
        self.data_test_list = []
        self.data_train_folds = []
        self.data_valid_folds = []
        self.sim_matrix = None
        self.default_split = 0.7

    def main(self, train_test, cv, complete):
        data_path = join(DATA_DIR, DATA_HEADER, DATA_HEADER + ".csv")
        train_data_path = join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_train.csv")
        test_data_path = join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_test.csv")
        fold_path = [join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_fold_" + str(i + 1) + ".csv") for i in
                     range(5)]

        if train_test:
            data_train = self.load_data(train_data_path)
            data_test = self.load_data(test_data_path)
            # data_train, data_test, self.no_features = select_features(data_train, data_test)
            data_complete = pd.concat([data_train, data_test])

            self.data_train_list = self.format_data(data_train)
            self.data_test_list = self.format_data(data_test)
        elif cv:
            data_complete = self.cross_validation_folds(fold_path)
        elif complete:
            data_complete = self.load_data(data_path)
            data_train, data_test = self.train_test_split(data_complete)
            # data_train, data_test, self.no_features = select_features(data_train, data_test)
            # data_complete = pd.concat([data_train, data_test])
            self.data_train_list = self.format_data(data_train)
            self.data_test_list = self.format_data(data_test)
        else:
            print('Error: No data file specified')
            return
        self.characterize_features(data_complete)
        self.characterize_labels(data_complete)
        if GET_MLD_PROP:
            self.multilabel_properties(data_complete)
        # plot_graph({0: list(range(NO_LABELS))}, self.sim_matrix, self.label_ref)

    def load_data(self, path):
        try:
            drop_index = []
            data = pd.read_csv(path)
            label_set_list = []
            try:
                data.set_index('ID', drop=True, inplace=True)
                self.id = True
            except KeyError:
                pass
            for idx, row in data.iterrows():
                label = [int(l) for l in row[NO_FEATURES:]]
                label_set = set([idx for idx, val in enumerate(label) if val == 1])
                label_set_list.append(label_set)
                if label_set.__len__() < 1:  # removes samples with no label
                    drop_index.append(idx)
            X = data.iloc[:, :NO_FEATURES]
            X_stand = (X - X.mean())/X.std()
            data.iloc[:, :NO_FEATURES] = X_stand
            data['labelset'] = label_set_list
            data.drop(drop_index, axis=0, inplace=True)
            return data
        except FileNotFoundError as fnferror:
            print(fnferror)

    # characterize features
    def characterize_features(self, data_complete):
        for dtype in data_complete.iloc[:, :self.no_features].dtypes:
            if dtype == "float64":
                self.dtypes.append(1)
            else:
                self.dtypes.append(0)
        dtypes = data_complete.iloc[:, :self.no_features].dtypes
        for (it, dtype) in enumerate(dtypes):
            if dtype == "int64":
                self.attribute_info.append(0)
            elif dtype == "float64":
                self.attribute_info.append([data_complete.iloc[:, it].min(),
                                            data_complete.iloc[:, it].max()])
        self.cov_inv = inv(cov(data_complete.iloc[:, :self.no_features].values.T))

    # characterize classes
    def characterize_labels(self, data_complete):
        self.label_count = NO_LABELS
        self.label_ref = {k: v for k, v in enumerate(data_complete.columns[self.no_features:-1])}
        label_matrix = data_complete.iloc[:, self.no_features:-1]
        label_matrix_sparse = csr_matrix(array(label_matrix).transpose())
        self.sim_matrix = cosine_similarity(label_matrix_sparse)

    # ِ multi-label properties
    def multilabel_properties(self, data_complete):
        count = sum([len(label) for label in data_complete['labelset']])
        lp_dict = {}
        for label in data_complete['labelset']:
            if str(label) in lp_dict.keys():
                lp_dict[str(label)] += 1
            else:
                lp_dict[str(label)] = 1
        self.distinct_lp_count = lp_dict.__len__()
        self.imbalance_lp = max(lp_dict.values()) / min(lp_dict.values())
        self.card = count / data_complete.__len__()
        self.density = self.card / self.label_count
        counts = [data_complete[classs].sum() for classs in data_complete.columns[self.no_features:-1]]
        class_count = dict(zip(list(data_complete.columns)[self.no_features:-1], counts))
        class_pi = [val / data_complete.__len__() for val in list(class_count.values())]
        imbalance_label = [max(class_pi) / val for val in class_pi]
        self.imbalance_mean = sum(imbalance_label) / self.label_count
        temp = [(val - self.imbalance_mean) ** 2 for val in imbalance_label]
        imbalance_label_sigma = sqrt(sum(temp) / (self.label_count - 1))
        self.cvir = imbalance_label_sigma / self.imbalance_mean
        plot_bar(class_count, 'frequency')
        plot_heatmap(self.sim_matrix, self.label_ref)
        self.class_ratio = {k: v/data_complete.__len__() for k, v in enumerate(class_count.values())}

        label_list = []
        if self.data_train_list:
            for row in self.data_train_list:
                if row[1] in label_list:
                    pass
                else:
                    label_list.append(row[1])
            for row in self.data_test_list:
                if row[1] in label_list:
                    pass
                else:
                    self.unseen_test_lp.append(row[1])
        test_labels = set.union(*[row[1] for row in self.data_test_list])
        self.unseen_test_labels = test_labels.difference(set.union(*[row[1] for row in self.data_train_list]))
        self.print_mldp()

    def print_mldp(self):
        print('Multi-label stats:')
        print('Training/Test samples: {} / {}'.format(self.data_train_list.__len__(), self.data_test_list.__len__()))
        print('Lcard: %.4f' % self.card)
        print('Ldens: %.4f' % self.density)
        print('Unseen test LP: %d' % self.unseen_test_lp.__len__())
        if self.unseen_test_labels:
            print('Unseen test labels: ', self.unseen_test_labels)

    # train-test split
    def train_test_split(self, data_complete):
        data_train, data_test = train_test_split(data_complete,
                                                 test_size=1 - self.default_split, random_state=SEED_NUMBER)
        return [data_train, data_test]

    # cross-validation data
    def cross_validation_folds(self, data_folds):
        k_fold = len(data_folds)
        fold_list = [self.load_data(name) for name in data_folds]
        for i in range(k_fold):
            frames = [fold_list[j] for j in range(k_fold) if j != i]
            train_fold = pd.concat(frames)
            valid_fold = fold_list[i]
            train_fold_list = self.format_data(train_fold)
            valid_fold_list = self.format_data(valid_fold)
            self.data_train_folds.append(train_fold_list)
            self.data_valid_folds.append(valid_fold_list)
        return pd.concat(fold_list)

    # format data
    def format_data(self, data):
        data_list = []
        data.sample(frac=1.0, random_state=SEED_NUMBER)
        if self.id:
            for idx, row in data.iterrows():
                data_list.append([list(row[:self.no_features]), row[-1], idx])
        else:
            for idx, row in data.iterrows():
                data_list.append([list(row[:self.no_features]), row[-1]])
        return data_list


if __name__ == "__main__":
    data_path0 = join(DATA_DIR, DATA_HEADER, DATA_HEADER + ".csv")
    train_data_path0 = join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_train.csv")
    test_data_path0 = join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_test.csv")
    fold_path0 = [join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_fold_" + str(i + 1) + ".csv") for i in range(5)]

    preprocessing = Preprocessing()
    preprocessing.main(0, 0, 1)
