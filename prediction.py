# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
from random import randint

from classifier import Classifier
from config import *
# from classifier_set import ClassifierSets


class Prediction:
    def __init__(self, popset, matchset):
        self.decision = set()
        self.vote = {}
        self.popset = popset
        self.matchset = matchset

    def max_prediction(self):
        tiebreak_numerosity = {}

        def update_value(lp, ref):
            if self.vote.get(tuple(lp)):
                self.vote[tuple(lp)] += (1 - self.popset[ref].loss) * self.popset[ref].numerosity
                tiebreak_numerosity[tuple(lp)] += self.popset[ref].numerosity
            else:
                self.vote[tuple(lp)] = (1 - self.popset[ref].loss) * self.popset[ref].numerosity
                tiebreak_numerosity[tuple(lp)] = self.popset[ref].numerosity

        [update_value(self.popset[ref].prediction, ref) for ref in self.matchset]
        max_vote = max(self.vote.values())

        if max_vote == 0:
            self.decision = set(list(self.vote.keys())[randint(0, self.vote.keys().__len__())])
            return
        candidate_lp = [lp for lp, v in self.vote.items() if v == max_vote]
        if candidate_lp.__len__() > 1:
            max_numerosity = max(tiebreak_numerosity.values())
            candidate_lp = [lp for lp in candidate_lp if tiebreak_numerosity[lp] == max_numerosity]
            if candidate_lp.__len__() > 1:
                self.decision = set(candidate_lp[randint(0, candidate_lp.__len__())])
            else:
                self.decision = set(candidate_lp[0])
        else:
            self.decision = set(candidate_lp[0])

    def aggregate_prediction(self):
        predicted_labels = set().union(*[self.popset[ref].prediction for ref
                                    in self.matchset])
        self.vote = dict.fromkeys(predicted_labels)

        def update_value(label, ref):
            if self.vote[label]:
                self.vote[label] += (1 - self.popset[ref].loss)*self.popset[ref].numerosity
            else:
                self.vote[label] = (1 - self.popset[ref].loss)*self.popset[ref].numerosity
        [update_value(label, ref) for ref in self.matchset for label in self.popset[ref].prediction]
        try:
            max_vote = max(self.vote.values())
            self.vote = {k: v/max_vote for k, v in self.vote.items()}
        except ZeroDivisionError:
            pass

    def one_threshold(self):
        [self.decision.add(label) for label in self.vote.keys() if self.vote[label] >= THETA]

    def rank_cut(self):
        labels_sorted = list({k: v for k, v in sorted(self.vote.items(), key=lambda item: item[1], reverse=True)}.keys())
        self.decision = set(labels_sorted[0:RANK_CUT])

    def get_prediction(self):
        return self.decision

# p-cut

if __name__ == "__main__":
    pop = [Classifier(1, 0, [0.5, 0.5], {1, 2}), Classifier(1, 0, [0.5, 0.5], {1, 3})]
    matchset = [0, 1]


