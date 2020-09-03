# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
from config import *


class Prediction:
    def __init__(self):
        self.prediction = set()
        self.vote = {}

    def max_prediction(self, popset, matchset, randint_func):
        tiebreak_numerosity = {}
        self.prediction = set()
        self.vote = {}

        def update_value(lp, ref):
            if self.vote.get(tuple(lp)):
                self.vote[tuple(lp)] += popset[ref].fitness * popset[ref].numerosity
                tiebreak_numerosity[tuple(lp)] += popset[ref].numerosity
            else:
                self.vote[tuple(lp)] = popset[ref].fitness * popset[ref].numerosity
                tiebreak_numerosity[tuple(lp)] = popset[ref].numerosity

        [update_value(popset[ref].prediction, ref) for ref in matchset]
        max_vote = max(self.vote.values())

        if max_vote == 0:
            [self.prediction.add(label) for label in list(self.vote.keys())
                [randint_func(0, self.vote.keys().__len__() - 1)]]
            return
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

    def aggregate_prediction(self, popset, matchset):
        self.prediction = set()
        self.vote = {}

        predicted_labels = set().union(*[popset[ref].prediction for ref
                                         in matchset])
        self.vote = dict.fromkeys(predicted_labels)
        numerosity = dict.fromkeys(predicted_labels)

        def update_value(label, ref):
            if self.vote[label]:
                self.vote[label] += popset[ref].fitness * popset[ref].numerosity
                numerosity[label] += popset[ref].numerosity
            else:
                self.vote[label] = popset[ref].fitness * popset[ref].numerosity
                numerosity[label] = popset[ref].numerosity

        [update_value(label, ref) for ref in matchset for label in popset[ref].prediction]
        try:
            # max_vote = max(self.vote.values())
            self.vote = {k: v / numerosity[k] for k, v in self.vote.items()}
        except (ZeroDivisionError, ValueError):
            pass

    def one_threshold(self):
        [self.prediction.add(label) for label in self.vote.keys() if self.vote[label] >= THETA]
        return [self.prediction, self.vote]

    def rank_cut(self):
        labels_sorted = list(
            {k: v for k, v in sorted(self.vote.items(), key=lambda item: item[1], reverse=True)}.keys())
        self.prediction = set(labels_sorted[0:RANK_CUT])
        return [self.prediction, self.vote]

    def get_prediction(self):
        return self.prediction


# p-cut


if __name__ == "__main__":
    # pop = [Classifier(1, 0, [0.5, 0.5], {1, 2}), Classifier(1, 0, [0.5, 0.5], {1, 3})]
    print('nothing here')
