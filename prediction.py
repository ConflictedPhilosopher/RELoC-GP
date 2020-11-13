# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
from numpy import argmax, zeros, sqrt
from sklearn.metrics import roc_curve

from config import *


class Prediction:
    def __init__(self):
        self.prediction = set()
        self.vote = {}
        self.theta = [0.5] * NO_LABELS

    def max_prediction(self, matching_cls, randint_func):
        tiebreak_numerosity = {}
        self.prediction = set()
        self.vote = {}

        def update_value(cl):
            lp = cl.prediction
            if self.vote.get(tuple(lp)):
                self.vote[tuple(lp)] += cl.fitness * cl.numerosity
                tiebreak_numerosity[tuple(lp)] += cl.numerosity
            else:
                self.vote[tuple(lp)] = cl.fitness * cl.numerosity
                tiebreak_numerosity[tuple(lp)] = cl.numerosity

        [update_value(cl) for cl in matching_cls]
        max_vote = max(self.vote.values())

        if max_vote == 0:
            [self.prediction.add(label) for label in list(self.vote.keys())
                [randint_func(0, self.vote.keys().__len__() - 1)]]
            return self.prediction
        candidate_lp = [lp for lp, v in self.vote.items() if v == max_vote]
        if candidate_lp.__len__() > 1:
            max_numerosity = max([tiebreak_numerosity[lp] for lp in candidate_lp])
            candidate_lp = [lp for lp in candidate_lp if tiebreak_numerosity[lp] == max_numerosity]
            if candidate_lp.__len__() > 1:
                [self.prediction.add(label) for label in candidate_lp[randint_func(0, candidate_lp.__len__() - 1)]]
            else:
                [self.prediction.add(label) for label in candidate_lp[0]]
        else:
            [self.prediction.add(label) for label in candidate_lp[0]]
        return self.prediction

    def aggregate_prediction(self, matching_cls):
        self.prediction = set()
        self.vote = {}

        predicted_labels = set().union(*[cl.prediction for cl in matching_cls])
        self.vote = dict.fromkeys(predicted_labels, 0.0)
        numerosity = dict.fromkeys(predicted_labels, 0)

        def update_value2(cl):
            if sum(cl.label_based.values()) > 0:
                label_acc = cl.label_based
            else:
                label_acc = {k: cl.fitness for k in cl.prediction}
            for label in cl.prediction:
                self.vote[label] += label_acc[label]
                numerosity[label] += cl.numerosity

        [update_value2(cl) for cl in matching_cls]
        try:
            # max_vote = max(self.vote.values())
            # self.vote = {k: v / numerosity[k] for k, v in self.vote.items()}
            self.vote = {k: v / matching_cls.__len__() for k, v in self.vote.items()}
            return self.vote
        except (ZeroDivisionError, ValueError):
            pass

    def optimize_theta(self, votes, targets):
        self.theta = []
        vote_list = zeros((votes.__len__(), NO_LABELS))
        target_list = zeros((votes.__len__(), NO_LABELS))
        idx = 0
        for vote, target in zip(votes, targets):
            for k, v in vote.items():
                vote_list[idx][k] = v
            for t in target:
                target_list[idx][t] = 1
            idx += 1
        for l in range(NO_LABELS):
            fpr, tpr, thresholds = roc_curve(target_list[:, l], vote_list[:, l])
            g_means = sqrt(tpr * (1 - fpr))
            self.theta.append(thresholds[argmax(g_means)])

    def one_threshold(self, vote):
        [self.prediction.add(label) for label in vote.keys() if vote[label] >= self.theta[label]]
        return self.prediction

    def rank_cut(self, vote):
        labels_sorted = list(
            {k: v for k, v in sorted(vote.items(), key=lambda item: item[1], reverse=True)}.keys())
        self.prediction = set(labels_sorted[0:RANK_CUT])
        return self.prediction

    def get_prediction(self):
        return self.prediction


# p-cut


if __name__ == "__main__":
    # pop = [Classifier(1, 0, [0.5, 0.5], {1, 2}), Classifier(1, 0, [0.5, 0.5], {1, 3})]
    print('nothing here')
