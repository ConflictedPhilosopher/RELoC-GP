# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
from numpy import argmax, zeros, sqrt, isnan, all, nan_to_num
from sklearn.metrics import roc_curve, precision_recall_curve

from config import *


def max_prediction(matching_cls, randint_func):
    tiebreak_numerosity = {}
    prediction = set()
    vote = {}

    def update_value(cl):
        lp = cl.prediction
        if vote.get(tuple(lp)):
            vote[tuple(lp)] += cl.fitness * cl.numerosity
            tiebreak_numerosity[tuple(lp)] += cl.numerosity
        else:
            vote[tuple(lp)] = cl.fitness * cl.numerosity
            tiebreak_numerosity[tuple(lp)] = cl.numerosity

    [update_value(cl) for cl in matching_cls]
    max_vote = max(vote.values())

    if max_vote == 0:
        [prediction.add(label) for label in list(vote.keys())
            [randint_func(0, vote.keys().__len__() - 1)]]
        return prediction
    candidate_lp = [lp for lp, v in vote.items() if v == max_vote]
    if candidate_lp.__len__() > 1:
        max_numerosity = max([tiebreak_numerosity[lp] for lp in candidate_lp])
        candidate_lp = [lp for lp in candidate_lp if tiebreak_numerosity[lp] == max_numerosity]
        if candidate_lp.__len__() > 1:
            [prediction.add(label) for label in candidate_lp[randint_func(0, candidate_lp.__len__() - 1)]]
        else:
            [prediction.add(label) for label in candidate_lp[0]]
    else:
        [prediction.add(label) for label in candidate_lp[0]]
    return prediction


def aggregate_prediction(matching_cls):
    vote = {}

    predicted_labels = set().union(*[cl.prediction for cl in matching_cls])
    vote = dict.fromkeys(predicted_labels, 0.0)
    numerosity = dict.fromkeys(predicted_labels, 0)

    def update_value(cl):
        if sum(cl.label_based.values()) > 0:
            label_acc = cl.label_based
        else:
            label_acc = {k: cl.fitness for k in cl.prediction}
        for label in cl.prediction:
            vote[label] += label_acc[label]
            numerosity[label] += 1

    [update_value(cl) for cl in matching_cls]
    try:
        max_vote = max(vote.values())
        vote = {k: v / numerosity[k] for k, v in vote.items()}
        # vote = {k: v / max_vote for k, v in vote.items()}
        return vote
    except (ZeroDivisionError, ValueError):
        pass


def optimize_theta(votes, targets):
    theta = []
    for l in range(NO_LABELS):
        if targets[:, l].nnz == 0:
            theta.append(1.0)
        else:
            precision, recall, thresholds = precision_recall_curve(targets[:, l].toarray(), votes[:, l].toarray())
            fscore = nan_to_num((2 * precision * recall) / (precision + recall))
            theta.append(thresholds[argmax(fscore)])
        # fpr, tpr, thresholds = roc_curve(target_list[:, l], vote_list[:, l])
        # if all(isnan(tpr)):
        #     theta.append(1.0)
        # else:
        #     g_means = sqrt(tpr * (1 - fpr))
        #     t = thresholds[argmax(g_means)]
        #     if t > 1:
        #         t = thresholds[argmax(g_means) + 1]
        #     theta.append(t)
    return theta


def one_threshold(vote, theta=None):
    if not theta:
        theta = [THETA] * NO_LABELS
    prediction = set()
    [prediction.add(label) for label in vote.keys() if vote[label] >= theta[label]]
    return prediction


def rank_cut(vote):
    labels_sorted = list(
        {k: v for k, v in sorted(vote.items(), key=lambda item: item[1], reverse=True)}.keys())
    return set(labels_sorted[0:RANK_CUT])


# p-cut


if __name__ == "__main__":
    # pop = [Classifier(1, 0, [0.5, 0.5], {1, 2}), Classifier(1, 0, [0.5, 0.5], {1, 3})]
    print('nothing here')
