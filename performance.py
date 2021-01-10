# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import operator
from sklearn.metrics import roc_curve, auc, coverage_error, label_ranking_average_precision_score
from numpy import zeros, isnan

from config import *


def exact_match(prediction, target):
    if prediction == target:
        return 1
    else:
        return 0


def precision(prediction, target):
    try:
        return prediction.intersection(target).__len__() / prediction.__len__()
    except ZeroDivisionError:
        return 0.0


def recall(prediction, target):
    try:
        return prediction.intersection(target).__len__() / target.__len__()
    except ZeroDivisionError:
        return 0.0


def accuracy(prediction, target):
    try:
        return prediction.intersection(target).__len__()/prediction.union(target).__len__()
    except ZeroDivisionError:
        return 0.0


def fscore(prediction, target):
    try:
        return 2 * prediction.intersection(target).__len__()\
                              / (prediction.__len__() + target.__len__())
    except ZeroDivisionError:
        return 0.0


def hamming_loss(prediction, target, n_labels):
    return prediction.symmetric_difference(target).__len__() / n_labels


def rank_loss(vote, target, n_labels):
    if not vote:
        return 1.0

    target_complement = set(range(0, n_labels)).difference(target)
    loss = 0
    for tc in target_complement:
        for t in target:
            if vote.get(t, 0) < vote.get(tc, -1e-5):
                loss += 1
    try:
        return loss / (target.__len__() * target_complement.__len__())
    except ZeroDivisionError:
        return 0.0


def one_error(vote, target):
    try:
        labels_max_vote = {max(vote.items(), key=operator.itemgetter(1))[0]}
    except ValueError:
        labels_max_vote = set()
    if labels_max_vote.intersection(target).__len__() > 0:
        return 0
    else:
        return 1.0


def coverage(vote, target, no_labels):
    vote0 = zeros(no_labels)
    target0 = zeros(no_labels)
    for k, v in vote.items():
        vote0[k] = v
    for t in target:
        target0[t] = 1.0
    vote0 = vote0.reshape((1, no_labels))
    target0 = target0.reshape((1, no_labels))
    return coverage_error(target0, vote0)


def rank_precision(vote, target, no_labels):
    vote0 = zeros(no_labels)
    target0 = zeros(no_labels)
    for k, v in vote.items():
        vote0[k] = v
    for t in target:
        target0[t] = 1.0
    vote0 = vote0.reshape((1, no_labels))
    target0 = target0.reshape((1, no_labels))
    rp = label_ranking_average_precision_score(target0, vote0)
    return rp


class Performance:
    def __init__(self):
        self.n_labels = NO_LABELS
        self.exact_match_example = 0.0
        self.precision_example = 0.0
        self.recall_example = 0.0
        self.accuracy_example = 0.0
        self.fscore_example = 0.0
        self.hamming_loss_example = 0.0
        self.rank_loss_example = 0.0
        self.one_error_example = 0.0
        self.class_based_measure = [dict(TP=0.0, FP=0.0, TN=0.0, FN=0.0)] * self.n_labels
        self.micro_precision = 0.0
        self.micro_recall = 0.0
        self.micro_fscore = 0.0
        self.macro_precision = 0.0
        self.macro_recall = 0.0
        self.macro_fscore = 0.0
        self.roc_auc = 0.0
        self.coverage_example = 0.0
        self.rank_precision_example = 0.0

    def update_example_based(self, vote, prediction, target):
        self.exact_match_example += exact_match(prediction, target)
        self.hamming_loss_example += hamming_loss(prediction, target, self.n_labels)
        self.precision_example += precision(prediction, target)
        self.recall_example += recall(prediction, target)
        self.fscore_example += fscore(prediction, target)
        self.accuracy_example += accuracy(prediction, target)
        if PREDICTION_METHOD == 2:
            self.one_error_example += one_error(vote, target)
            self.rank_loss_example += rank_loss(vote, target, self.n_labels)
            self.coverage_example += coverage(vote, target, self.n_labels)
            self.rank_precision_example += rank_precision(vote, target, self.n_labels)

    def update_class_based(self, prediction, target):
        tp = target.intersection(prediction)
        fp = prediction.difference(target.intersection(prediction))
        fn = target.difference(target.intersection(prediction))

        def update_single(label, where):
            class_dict = self.class_based_measure[label].copy()
            class_dict[where] += 1
            self.class_based_measure[label] = class_dict
        [update_single(label, 'TP') for label in tp]
        [update_single(label, 'FP') for label in fp]
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
        self.macro_precision = sum([class_dict['TP'] / (class_dict['TP'] + class_dict['FP'] + 1e-3) for
                                    class_dict in self.class_based_measure]) / self.n_labels
        self.macro_recall = sum([class_dict['TP'] / (class_dict['TP'] + class_dict['FN'] + 1e-3) for
                                class_dict in self.class_based_measure]) / self.n_labels
        self.macro_fscore = 2 * (self.macro_precision * self.macro_recall) \
            / (self.macro_precision + self.macro_recall + 1e-3)

    def roc(self, votes, targets):
        vote_list = zeros((votes.__len__(), self.n_labels))
        target_list = zeros((votes.__len__(), self.n_labels))
        idx = 0
        for vote, target in zip(votes, targets):
            for k, v in vote.items():
                vote_list[idx][k] = v
            for t in target:
                target_list[idx][t] = 1
            idx += 1

        roc_auc = []
        for l in range(self.n_labels):
            fpr, tpr, _ = roc_curve(target_list[:, l], vote_list[:, l])
            roc_auc.append(auc(fpr, tpr))
        roc_auc = [0 if isnan(val) else val for val in roc_auc]
        self.roc_auc = sum(roc_auc) / self.n_labels

    def get_report(self, sample_count):
        multi_label_perf = dict()
        multi_label_perf['em'] = self.exact_match_example / sample_count
        multi_label_perf['hl'] = self.hamming_loss_example / sample_count
        multi_label_perf['acc'] = self.accuracy_example / sample_count
        multi_label_perf['pr'] = self.precision_example / sample_count
        multi_label_perf['re'] = self.recall_example / sample_count
        multi_label_perf['f'] = self.fscore_example / sample_count
        multi_label_perf['micro-f'] = self.micro_fscore
        multi_label_perf['macro-f'] = self.macro_fscore
        multi_label_perf['micro-pr'] = self.micro_precision
        multi_label_perf['macro-pr'] = self.macro_precision
        multi_label_perf['micro-re'] = self.micro_recall
        multi_label_perf['macro-re'] = self.macro_recall
        multi_label_perf['1e'] = self.one_error_example / sample_count
        multi_label_perf['rl'] = self.rank_loss_example / sample_count
        multi_label_perf['cov-error'] = self.coverage_example / sample_count
        multi_label_perf['rank-pr'] = self.rank_precision_example / sample_count
        multi_label_perf['roc-auc'] = self.roc_auc
        return multi_label_perf

# extended hamming loss


if __name__ == "__main__":
    measure = Performance(5)
    prediction0 = {1, 2}
    target0 = {2, 4}
    measure.update_class_based(prediction0, target0)
    measure.micro_average()
