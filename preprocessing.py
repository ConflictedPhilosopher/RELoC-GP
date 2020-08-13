# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import os.path
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split

from config import *


class Preprocessing:
    def __init__(self):
        self.label_count = 0
        self.label_dict = {}
        self.distinct_lp = 0
        self.imbalance_lp = 0
        self.card = 0.0
        self.density = 0.0
        self.class_pi = []
        self.imbalance_mean = 0.0
        self.cvir = 0.0
        self.attribute_info = []
        self.dtypes = []
        self.data_train_list = []
        self.data_test_list = []
        self.data_train_count = 0
        self.data_test_count = 0
        self.data_complete_count = 0
        self.data_train_folds = []
        self.data_valid_folds = []
        self.default_split = 0.7

    def main(self,  train_test=None, cv=None, complete=None):
        data_path = os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + ".csv")
        train_data_path = os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_train.csv")
        test_data_path = os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_test.csv")
        fold_path = [os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_fold_" + str(i + 1) + ".csv") for i in
                     range(5)]

        if train_test:
            data_train = self.load_data(train_data_path)
            data_test = self.load_data(test_data_path)
            self.data_train_count = len(data_train)
            self.data_test_count = len(data_test)
            data_complete = pd.concat([data_train, data_test])

            self.data_train_list = self.format_data(data_train)
            self.data_test_list = self.format_data(data_test)
        elif cv:
            data_complete = self.cross_validation_folds(fold_path)
        elif complete:
            data_complete = self.load_data(data_path)
            data_train, data_test = self.train_test_split(data_complete)
            self.data_train_list = self.format_data(data_train)
            self.data_test_list = self.format_data(data_test)
        else:
            print('Error: No data file specified')
            return
        self.data_complete_count = data_complete.__len__()
        self.characterize_features(data_complete)
        self.characterize_labels(data_complete)
        if GET_MLD_PROP:
            self.multilabel_properties(data_complete)

# load data (.csv)
    def load_data(self, path):
        try:
            data = pd.read_csv(path)
            label_set_list = []
            for idx, row in data.iterrows():
                label = [int(l) for l in row[NO_FEATURES:]]
                label_set = set([idx for idx, val in enumerate(label) if val == 1])
                label_set_list.append(label_set)
            data['labelset'] = label_set_list
            return data
        except FileNotFoundError as fnferror:
            print(fnferror)

# feature selection

# characterize features
    def characterize_features(self, data_complete):
        for dtype in data_complete.iloc[:, :NO_FEATURES].dtypes:
            if dtype == "float64":
                self.dtypes.append(1)
            else:
                self.dtypes.append(0)
        dtypes = data_complete.iloc[:, :NO_FEATURES].dtypes
        for (it, dtype) in enumerate(dtypes):
            if dtype == "int64":
                self.attribute_info.append(0)
            elif dtype == "float64":
                self.attribute_info.append([data_complete.iloc[:, it].min(),
                                           data_complete.iloc[:, it].max()])

# characterize classes
    def characterize_labels(self, data_complete):
        self.label_count = len(data_complete.iloc[0, NO_FEATURES:-1])
        for label in data_complete['labelset']:
            if str(label) in self.label_dict.keys():
                self.label_dict[str(label)] += 1
            else:
                self.label_dict[str(label)] = 1
        self.distinct_lp = self.label_dict.__len__()
        self.imbalance_lp = max(self.label_dict.values()) \
            / min(self.label_dict.values())

# ِ multi-label properties
    def multilabel_properties(self, data_complete):
        count = sum([len(label) for label in data_complete['labelset']])
        self.card = count/self.data_complete_count
        self.density = self.card/self.label_count
        class_dict = dict(zip(range(self.label_count), [0]*self.label_count))
        for label in data_complete['labelset']:
            for lbl in label:
                class_dict[lbl] += 1
        self.class_pi = [val/self.data_complete_count for val in list(class_dict.values())]
        imbalance_label = [max(self.class_pi)/val for val in self.class_pi]
        self.imbalance_mean = sum(imbalance_label)/self.label_count
        temp = [(val - self.imbalance_mean)**2 for val in imbalance_label]
        imbalance_label_sigma = sqrt(sum(temp) / (self.label_count - 1))
        self.cvir = imbalance_label_sigma / self.imbalance_mean

# train-test split
    def train_test_split(self, data_complete):
        data_train, data_test = train_test_split(data_complete,
                                test_size=1-self.default_split, random_state=SEED_NUMBER)
        self.data_train_count = len(data_train)
        self.data_test_count = len(data_test)
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
        data_complete = pd.concat(fold_list)
        self.data_complete_count = data_complete.__len__()
        return data_complete

# format data
    def format_data(self, data):
        data_list = []
        # data.sample(frac=1.0, random_state=SEED_NUMBER)
        for idx, row in data.iterrows():
            data_list.append([list(row[:NO_FEATURES]), row[-1]])
        return data_list


if __name__ == "__main__":
    data_path0 = os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + ".csv")
    train_data_path0 = os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_train.csv")
    test_data_path0 = os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_test.csv")
    fold_path0 = [os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_fold_" + str(i+1) + ".csv") for i in range(5)]

    preprocessing = Preprocessing()
    preprocessing.main(0, 0, 1)
