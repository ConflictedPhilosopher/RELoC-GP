# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
from copy import deepcopy

from config import *


class Classifier:
    def __init__(self):
        self.specified_atts = []
        self.condition = []
        self.prediction = set()
        self.parent_prediction = []
        self.numerosity = 1
        self.match_count = 0
        self.correct_count = 0
        self.loss = 0.0
        self.fitness = INIT_FITNESS
        self.ave_matchset_size = 0
        self.init_time = 0
        self.ga_time = 0
        self.deletion_vote = 0.0

    def classifier_cover(self, set_size, it, state, target, attribute_info, dtypes, random_func):
        self.ga_time = it
        self.init_time = it
        self.ave_matchset_size = set_size
        self.prediction = target
        for ref, x in enumerate(state):
            if random_func.random() < (1 - PROB_HASH):
                self.specified_atts.append(ref)
                self.condition.append(self.build_match(x, attribute_info[ref], dtypes[ref], random_func))

    def build_match(self, x, att_info, dtype, random_func):
        if dtype:
            att_range = att_info[1] - att_info[0]
            radius = random_func.randint(15, 85) * 0.01 * (att_range / 2.0)
            return [x - radius, x + radius]
        elif not dtype:
            return x
        else:
            print("attribute type unidentified!")

    def classifier_copy(self, classifier_old, it):
        self.specified_atts = deepcopy(classifier_old.specified_atts)
        self.condition = deepcopy(classifier_old.condition)
        self.prediction = deepcopy(classifier_old.prediction)
        self.parent_prediction = deepcopy(classifier_old.parent_prediction)
        self.ave_matchset_size = classifier_old.ave_matchset_size
        self.init_time = it
        self.ga_time = it
        self.fitness = classifier_old.fitness
        self.loss = classifier_old.loss

    def classifier_reboot(self, classifier_info, dtypes):
        classifier_info = classifier_info.to_list()
        condition = classifier_info[:NO_FEATURES]

        def update_cond(ref, att_val):
            if dtypes[ref] == 1:
                self.specified_atts.append(ref)
                self.condition.append([float(x) for x in att_val.split(";")])
            else:
                self.specified_atts.append(ref)
                self.condition.append(int(att_val))

        for ref, att_val in enumerate(condition):
            if att_val == '#' or att_val == ' #':
                pass
            else:
                update_cond(ref, att_val)

        self.prediction = set(int(n) for n in classifier_info[NO_FEATURES].split(";"))
        self.fitness, self.loss, self.correct_count, self.numerosity, self.match_count, self.ave_matchset_size, \
            self.init_time, self.ga_time = classifier_info[NO_FEATURES + 1:]

    def update_numerosity(self, num):
        self.numerosity += num

    def update_correct(self):
        self.correct_count += 1

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
        self.loss += (self.prediction.symmetric_difference(target).__len__() / (float(NO_LABELS)))
        self.loss /= float(self.match_count)

        # update_fitness()
        self.fitness = max(pow(1 - self.loss, NU), INIT_FITNESS)

    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_loss(self, loss):
        self.loss = loss

if __name__ == "__main__":
    # classifier = Classifier(1, 0, [0.5, 0.5], {1, 2})
    # string = classifier.classifier_print()
    print('nothing goes here!')
