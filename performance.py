# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import operator
from sklearn.metrics import roc_curve, auc
import numpy as np

from config import *


class Performance:
    def __init__(self, label_count):
        self.label_count = label_count
        self.exact_match_example = 0.0
        self.precision_example = 0.0
        self.recall_example = 0.0
        self.accuracy_example = 0.0
        self.fscore_example = 0.0
        self.hamming_loss_example = 0.0
        self.rank_loss_example = 0.0
        self.one_error_example = 0.0
        self.class_based_measure = [dict(TP=0.0, FP=0.0, TN=0.0, FN=0.0)] * self.label_count
        self.micro_precision = 0.0
        self.micro_recall = 0.0
        self.micro_fscore = 0.0
        self.macro_precision = 0.0
        self.macro_recall = 0.0
        self.macro_fscore = 0.0
        self.roc_auc = 0.0

    def exact_match(self, prediction, target):
        if prediction == target:
            return 1
        else:
            return 0

    def precision(self, prediction, target):
        try:
            return prediction.intersection(target).__len__()/target.__len__()
        except ZeroDivisionError:
            return 0.0

    def recall(self, prediction, target):
        try:
            return prediction.intersection(target).__len__()/prediction.__len__()
        except ZeroDivisionError:
            return 0.0

    def accuracy(self, prediction, target):
        try:
            return prediction.intersection(target).__len__()/prediction.union(target).__len__()
        except ZeroDivisionError:
            return 0.0

    def fscore(self, prediction, target):
        try:
            return 2 * prediction.intersection(target).__len__()\
                                  / (prediction.__len__() + target.__len__())
        except ZeroDivisionError:
            return 0.0

    def hamming_loss(self, prediction, target):
        return prediction.symmetric_difference(target).__len__() / self.label_count

    def rank_loss(self, vote, target):
        if not vote:
            return 1.0

        target_complement = set(range(0, self.label_count)).difference(target)
        loss = 0
        for tc in target_complement:
            for t in target:
                if vote.get(t, 0) < vote.get(tc, -1e-5):
                    loss += 1
        try:
            return loss / (target.__len__() * target_complement.__len__())
        except ZeroDivisionError:
            return 0.0

    def one_error(self, vote, target):
        try:
            labels_max_vote = {max(vote.items(), key=operator.itemgetter(1))[0]}
        except ValueError:
            labels_max_vote = set()
        if labels_max_vote.intersection(target).__len__() > 0:
            return 0
        else:
            return 1.0

    def update_example_based(self, vote, prediction, target):
        self.exact_match_example += self.exact_match(prediction, target)
        self.hamming_loss_example += self.hamming_loss(prediction, target)
        self.precision_example += self.precision(prediction, target)
        self.recall_example += self.recall(prediction, target)
        self.fscore_example += self.fscore(prediction, target)
        self.accuracy_example += self.accuracy(prediction, target)
        if PREDICTION_METHOD is not 'max':
            self.one_error_example += self.one_error(vote, target)
            self.rank_loss_example += self.rank_loss(vote, target)

    def update_class_based(self, prediction, target):
        tp = target.intersection(prediction)
        fp = prediction.difference(target.intersection(prediction))
        tn = set(range(0, self.label_count)).difference(target).difference(prediction)
        fn = target.difference(target.intersection(prediction))

        def update_single(label, where):
            class_dict = self.class_based_measure[label].copy()
            class_dict[where] += 1
            self.class_based_measure[label] = class_dict
        [update_single(label, 'TP') for label in tp]
        [update_single(label, 'FP') for label in fp]
        [update_single(label, 'TN') for label in tn]
        [update_single(label, 'FN') for label in fn]

    def micro_average(self):
        tp_sum = sum([class_dict['TP'] for class_dict in self.class_based_measure])
        fp_sum = sum([class_dict['FP'] for class_dict in self.class_based_measure])
        fn_sum = sum([class_dict['FN'] for class_dict in self.class_based_measure])
        try:
            self.micro_precision = tp_sum / (tp_sum + fp_sum)
        except ZeroDivisionError:
            self.micro_precision = 0.0
        try:
            self.micro_recall = tp_sum / (tp_sum + fn_sum)
        except ZeroDivisionError:
            self.micro_recall = 0.0
        self.micro_fscore = 2 * (self.micro_precision * self.micro_recall) \
            / (self.micro_precision + self.micro_recall + 1e-3)

    def macro_average(self):
        self.macro_precision = sum([class_dict['TP'] / (class_dict['TP'] + class_dict['FP'] + 1) for
                                    class_dict in self.class_based_measure]) / self.label_count
        self.macro_recall = sum([class_dict['TP'] / (class_dict['TP'] + class_dict['FN'] + 1) for
                                class_dict in self.class_based_measure]) / self.label_count
        self.macro_fscore = 2 * (self.macro_precision * self.macro_recall) \
            / (self.macro_precision + self.macro_recall + 1e-3)

    def roc(self, vote_list, target_list):
        roc_auc = []
        fpr = dict()
        tpr = dict()
        for l in range(2):
            fpr[l], tpr[l], _ = roc_curve(target_list[:, l], vote_list[:, l])
            roc_auc[l] = auc(fpr[l], tpr[l])
        where_are_NaNs = np.isnan(roc_auc)
        roc_auc[where_are_NaNs] = 0.0
        self.roc_auc = sum(roc_auc) / self.label_count

# extended hamming loss


if __name__ == "__main__":
    measure = Performance(5)
    prediction0 = {1, 2}
    target0 = {2, 4}
    measure.update_class_based(prediction0, target0)
    measure.micro_average()
