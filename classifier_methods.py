# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
from random import randint
from preprocessing import Preprocessing
from classifier import Classifier
from config import *


class ClassifierMethods:
    def __init__(self):
        self.preprocessing = Preprocessing()

    def build_match(self, ref, x):
        att_info = self.preprocessing.attribute_info[ref]
        dtype = self.preprocessing.dtypes[ref]
        "continuous attribute"
        if dtype == "float64":
            att_range = att_info[1] - att_info[0]
            radius = randint(25, 75) * 0.01 * (att_range / 2.0)
            return [x - radius, x + radius]
        elif dtype == "int64":
            return x
        else:
            print("attribute type unidentified!")

    def match(self, classifier, state):
        for ref in classifier.specified_atts:
            x = state[ref]
            if self.preprocessing.dtypes[ref] == "float64":
                if classifier.condition[ref][0] < x < classifier.condition[ref][1]:
                    pass
                else:
                    return False
            else:
                if x == classifier.condition[ref]:
                    pass
                else:
                    return False
        return True

    def get_deletion_vote(self, classifier, ave_fitness):
        delta = 0.5   #TODO replace from constants
        theta_del = 0.5   #TODO replace from constants
        init_fitness = 0.01  #TODO replace from constants
        if classifier.fitness >= ave_fitness * delta or classifier.match_count < theta_del:
            classifier.deletion_vote = classifier.ave_matchset_size * classifier.numerosity
        elif classifier.fitness == init_fitness:
            classifier.deletion_vote = classifier.ave_matchset_size * classifier.numerosity * ave_fitness / \
                                       (init_fitness / classifier.numerosity)
        else:
            classifier.deletion_vote = classifier.ave_matchset_size * classifier.numerosity * ave_fitness / \
                                       (classifier.fitness / classifier.numerosity)

    def is_equal(self, classifier1, classifier2):
        if classifier1.prediction == classifier2.prediction and \
                len(classifier1.specified_atts) == len(classifier2.specified_atts):
            if classifier1.specified_atts.sort() == classifier2.specified_atts.sort():
                for ref in classifier1.specified_atts:
                    if classifier1.condition[ref] == classifier2.condition[ref]:
                        pass
                    else:
                        return False
                return True
        return False

    def subsumption(self, classifier1, classifier2):
        if classifier1.prediction == classifier2.prediction:
            if self.is_subsumer(classifier1) and self.is_more_general(classifier1, classifier2):
                return True
        return False

    def is_subsumer(self, classifier1):
        if classifier1.match_count > THETA_SUB and classifier1.loss < LOSS_SUB:
            return True
        return False

    def is_more_general(self, classifier1, classifier2):
        if len(classifier1.specified_atts) > len(classifier2.specified_atts):
            return False
        for ref in classifier1.specified_atts:
            if ref not in classifier2.specified_atts:
                return False
            if self.preprocessing.dtypes[ref] == "float64":
                if classifier1.condition[ref][0] < classifier2.condition[ref][0]:
                    return False
                if classifier1.condition[ref][1] > classifier2.condition[ref][1]:
                    return False

        return True


if __name__ == "__main__":
    classifier10 = Classifier(1, 0, [0.5, 0.5], {1, 2})
    classifier20 = Classifier(1, 0, [0.25, 0.25], {1, 2, 3})
    cl_method = ClassifierMethods()
    cl_method.is_subsumer(classifier10)
    cl_method.subsumption(classifier10, classifier20)
