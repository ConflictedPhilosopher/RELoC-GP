# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------

from random import random
from copy import deepcopy
from classifier_methods import ClassifierMethods
from config import *
import environment as env


class Classifier:
    def __init__(self, a=None, b=None, c=None, d=None):
        self.specified_atts = []
        self.condition = []
        self.prediction = {}
        self.parent_prediction = []
        self.numerosity = 1
        self.match_count = 1
        self.correct_count = 1
        self.loss = 0.0
        self.fitness = INIT_FITNESS
        self.ave_matchset_size = 0
        self.init_time = 0
        self.ga_time = 0
        self.deletion_vote = 0.0

        if isinstance(c, list):
            self.classifier_cover(a, b, c, d)
        elif isinstance(a, Classifier):
            self.classifier_copy(a, b)
        elif isinstance(a, list) and not b:
            self.classifier_reboot(a)
        else:
            print("Classifier: Error building classifier.")

    def classifier_cover(self, set_size, it, state, target):
        self.ga_time = it
        self.init_time = it
        self.ave_matchset_size = set_size
        self.prediction = target
        method = ClassifierMethods()

        for ref, x in enumerate(state):
            if random() < (1 - PROB_HASH):
                self.specified_atts.append(ref)
                self.condition.append(method.build_match(ref, x))

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
        self.correct_count = classifier_old.correct_count

    def classifier_reboot(self, classifier_info):
        for ref in range(NO_FEATURES):
            if classifier_info[ref] != "#":
                if env.preprocessing.dtypes[ref] == "float64":
                    self.specified_atts.append(ref)
                    self.condition.append(list(classifier_info[ref].split(";")))
                else:
                    self.specified_atts.append(ref)
                    self.condition.append(classifier_info[ref])

        self.prediction = set(int(n) for n in classifier_info[NO_FEATURES].split(";"))
        i = 1
        self.fitness = float(classifier_info[NO_FEATURES + i])
        i += 1
        self.loss = float(classifier_info[NO_FEATURES + i])
        i += 1
        self.correct_count = int(classifier_info[NO_FEATURES + i])
        i += 1
        self.numerosity = int(classifier_info[NO_FEATURES + i])
        i += 1
        self.correct_count = int(classifier_info[NO_FEATURES + i])
        i = + 1
        self.ave_matchset_size = float(classifier_info[NO_FEATURES + i])
        i += 1
        self.init_time = int(classifier_info[NO_FEATURES + i])
        i += 1
        self.ga_time = int(classifier_info[NO_FEATURES + i])

    def classifier_print(self):
        classifier_string = ""
        for ref in range(NO_FEATURES):
            if ref in self.specified_atts:
                ind = self.specified_atts.index(ref)
                if env.preprocessing.dtypes[ref] == "float64":
                    classifier_string += (str("%.4f" % self.condition[ind][0]) + ';'
                                          + str("%.4f" % self.condition[ind][1]))
                else:
                    classifier_string += str(self.condition[ind])
            else:
                classifier_string += "#"
            classifier_string += "\t"
        prediction_string = ";".join([str(label) for label in self.prediction])
        classifier_string += (prediction_string + "\t")
        parameter_string = str("%.4f" % self.fitness) + "\t" + \
                           str("%.4f" % self.loss) + "\t" + \
                           str("%d" % self.correct_count) + "\t" + \
                           str("%d" % self.numerosity) + "\t" + \
                           str("%d" % self.match_count) + "\t" + \
                           str("%.4f" % self.ave_matchset_size) + "\t" + \
                           str("%d" % self.init_time) + "\t" + \
                           str("%d" % self.ga_time) + "\t"
        classifier_string += parameter_string
        return classifier_string

    def update_experience(self):
        self.match_count += 1

    def update_matchset_size(self, m_size):
        if self.match_count < 1.0 / BETA:
            self.ave_matchset_size += (m_size - self.ave_matchset_size) / float(self.match_count)
        else:
            self.ave_matchset_size += BETA * (m_size - self.ave_matchset_size)

    def update_numerosity(self, num):
        self.numerosity += num

    def update_correct(self):
        self.correct_count += 1

    def update_loss(self, target):
        self.loss += (self.prediction.symmetric_difference(target)/float(self.match_count))

    def update_ga_time(self, time):
        self.ga_time = time

    def update_fitness(self):
        self.fitness = max(pow(1 - self.loss, NU), INIT_FITNESS)

    def set_fitness(self, fitness):
        self.fitness = fitness

    def update_params(self, m_size, target):
        self.update_experience()
        self.update_matchset_size(m_size)
        self.update_loss(target)
        self.update_fitness()


if __name__ == "__main__":
    classifier = Classifier(1, 0, [0.5, 0.5], {1, 2})
    string = classifier.classifier_print()
