# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
from copy import deepcopy

from config import *


def build_match(x, att_info, dtype, random_func):
    if dtype:
        att_range = att_info[1] - att_info[0]
        radius_l = random_func.randint(25, 75) * 0.01 * (att_range / 2.0)
        radius_r = random_func.randint(25, 75) * 0.01 * (att_range / 2.0)
        return [max(att_info[0], (x - radius_l)), min(att_info[1], x + radius_r)]
    elif not dtype:
        return x
    else:
        print("attribute type unidentified!")


class Classifier:
    def __init__(self):
        self.specified_atts = []
        self.condition = []
        self.prediction = set()
        self.parent_prediction = []
        self.numerosity = 1
        self.match_count = 0
        self.loss = 0.0
        # self.fscore = 0.0
        self.label_based = {}
        self.fitness = INIT_FITNESS
        self.ave_matchset_size = 0
        self.init_time = 0
        self.ga_time = 0
        self.deletion_vote = 0.0

    def classifier_cover(self, set_size, it, state, target, attribute_info, dtypes, random_func):
        self.init_time = it
        self.ave_matchset_size = set_size
        self.prediction = target
        og = True
        while og:
            for ref, x in enumerate(state):
                if random_func.random() < (1 - PROB_HASH):
                    self.specified_atts.append(ref)
                    self.condition.append(build_match(x, attribute_info[ref], dtypes[ref], random_func))
                    og = False
        self.label_based = {k: 0.0 for k in self.prediction}

    def classifier_copy(self, classifier_old, it):
        self.specified_atts = deepcopy(classifier_old.specified_atts)
        self.condition = deepcopy(classifier_old.condition)
        self.prediction = deepcopy(classifier_old.prediction)
        self.parent_prediction = deepcopy(classifier_old.parent_prediction)
        self.ave_matchset_size = classifier_old.ave_matchset_size
        self.init_time = it
        self.ga_time = it
        self.fitness = classifier_old.fitness
        self.label_based = {k: 0.0 for k in self.prediction}

    def classifier_reboot(self, classifier_info, dtypes):
        classifier_info = classifier_info.to_list()
        condition = classifier_info[:NO_FEATURES]

        def update_cond(ref0, att_val0):
            if dtypes[ref0] == 1:
                self.specified_atts.append(ref0)
                self.condition.append([float(x) for x in att_val0.split(";")])
            else:
                self.specified_atts.append(ref0)
                self.condition.append(int(att_val0))

        for ref, att_val in enumerate(condition):
            if att_val == '#' or att_val == ' #':
                pass
            else:
                update_cond(ref, att_val)

        self.prediction = set(int(n) for n in classifier_info[NO_FEATURES+1].split(";"))
        label_precisions = classifier_info[NO_FEATURES + 2]
        self.label_based = {int(kv.split("%")[0]): float(kv.split("%")[1]) for kv in label_precisions.split(";")}
        self.fitness, self.loss, self.numerosity, self.match_count, self.ave_matchset_size, self.init_time, \
            self.ga_time = classifier_info[NO_FEATURES + 3:]

        # TODO parent prediction to be added

    def update_numerosity(self, num):
        self.numerosity += num

    def update_ga_time(self, time):
        self.ga_time = time

    def update_params(self, m_size, target):
        #  update_experience()
        self.match_count += 1

        # update_matchset_size(m_size)
        if self.match_count < 1.0 / BETA:
            self.ave_matchset_size += (m_size - self.ave_matchset_size) / float(self.match_count)
        else:
            self.ave_matchset_size += BETA * (m_size - self.ave_matchset_size)

        # update_loss(target)
        if not self.prediction.issubset(target):
            self.loss += (self.prediction.symmetric_difference(target).__len__() / NO_LABELS)

        # update f-score(target)
        # self.fscore += (2 * self.prediction.intersection(target).__len__() /
        #                 (self.prediction.__len__() + target.__len__()))

        # update_fitness()
        self.fitness = max((1 - self.loss / self.match_count) ** NU, INIT_FITNESS)
        # self.fitness = max(((accuracy_sum + self.fscore/self.match_count) / 2) ** NU, INIT_FITNESS)

    def set_fitness(self, fitness):
        self.fitness = fitness

    def estimate_label_based(self, target_labels):
        for k in self.prediction:
            count = [1 for labels in target_labels if k in labels]
            self.label_based[k] = sum(count)/target_labels.__len__()


if __name__ == "__main__":
    # classifier = Classifier(1, 0, [0.5, 0.5], {1, 2})
    # string = classifier.classifier_print()
    print('nothing goes here!')
