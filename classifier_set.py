# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import random
import timer

from classifier_methods import ClassifierMethods
from classifier import Classifier
from config import *


class ClassifierSets:
    def __init__(self, attribute_info, dtypes, timer):
        self.popset = []
        self.matchset = []
        self.correctset = []
        self.micro_pop_size = 0
        self.ave_generality = 0.0
        self.ave_loss = 0.0
        self.cl_methods = ClassifierMethods(dtypes, timer)
        self.attribute_info = attribute_info
        self.dtypes = dtypes
        self.timer = timer

    def make_matchset(self, state, target, it):

        def match(classifier, state0):
            for idx, ref in enumerate(classifier.specified_atts):
                x = state0[ref]
                if self.dtypes[ref]:
                    if classifier.condition[idx][0] < x < classifier.condition[idx][1]:
                        pass
                    else:
                        return False
                else:
                    if x == classifier.condition[idx]:
                        pass
                    else:
                        return False
            return True

        self.timer.start_matching()
        covering = True
        self.matchset = [ind for (ind, classifier) in enumerate(self.popset) if
                         match(classifier, state)]
        self.timer.stop_matching()
        numerosity_sum = sum([self.popset[ind].numerosity for ind in self.matchset])
        for ind in self.matchset:
            if self.popset[ind].prediction == target:
                covering = False
                return

        if covering:
            new_classifier = Classifier(self.attribute_info, self.dtypes, numerosity_sum + 1, it, state, target)
            self.insert_classifier_pop(new_classifier, True)
            self.matchset.append(self.popset.__len__() - 1)

    def make_eval_matchset(self, state):

        def match(classifier, state0):
            self.timer.start_subsumption()
            for idx, ref in enumerate(classifier.specified_atts):
                x = state0[ref]
                if self.dtypes[ref] == "float64":
                    if classifier.condition[idx][0] < x < classifier.condition[idx][1]:
                        pass
                    else:
                        return False
                else:
                    if x == classifier.condition[idx]:
                        pass
                    else:
                        return False
            self.timer.stop_subsumption()
            return True

        self.matchset = [ind for (ind, classifier) in enumerate(self.popset) if
                         match(classifier, state)]

    def make_correctset(self, target):
        self.correctset = [ind for ind in self.matchset if self.popset[ind].prediction == target]

# deletion methods
    def deletion(self):
        self.timer.start_deletion()
        while self.micro_pop_size > MAX_CLASSIFIER:
            self.delete_from_sets()
        self.timer.stop_deletion()

    def delete_from_sets(self):
        random.seed(SEED_NUMBER)
        ave_fitness = sum([classifier.fitness for classifier in self.popset])\
                       / float(self.micro_pop_size)
        vote_list = [self.cl_methods.get_deletion_vote(cl, ave_fitness) for cl in self.popset]
        vote_sum = sum(vote_list)
        choice_point = vote_sum * random.random()

        new_vote_sum = 0.0
        for idx in range(vote_list.__len__()):
            new_vote_sum += vote_list[idx]
            if new_vote_sum > choice_point:
                cl = self.popset[idx]
                cl.update_numerosity(-1)
                self.micro_pop_size -= 1
                if cl.numerosity < 1:
                    self.remove_from_pop(idx)
                    self.remove_from_matchset(idx)
                    self.remove_from_correctset(idx)
                return

    def remove_from_pop(self, ref):
        self.popset.pop(ref)

    def remove_from_matchset(self, ref):
        try:
            self.matchset.remove(ref)
            matchset_copy = [ind-1 for ind in self.matchset if ind > ref]
            self.matchset = matchset_copy
        except ValueError:
            pass

    def remove_from_correctset(self, ref):
        try:
            self.correctset.remove(ref)
            correctset_copy = [ind-1 for ind in self.correctset if ind > ref]
            self.correctset = correctset_copy
        except ValueError:
            pass

# run genetic algorithm
    def insert_classifier_pop(self, classifier, search_matchset=False):
        existing_classifier = self.get_identical(classifier, search_matchset)
        if isinstance(existing_classifier, Classifier):
            existing_classifier.update_numerosity(1)
        else:
            self.popset.append(classifier)
        self.micro_pop_size += 1

    def insert_discovered_classifier(self, offspring1, offspring2, parent1, parent2):
        if DO_SUBSUMPTION:
            self.timer.start_subsumption()
            if offspring1.specified_atts.__len__ > 0:
                self.subsume_into_parents(offspring1, parent1, parent2)
            if offspring2.specified_atts.__len__ > 0:
                self.subsume_into_parents(offspring2, parent1, parent2)
            self.timer.stop_subsumption()
        else:
            self.insert_classifier_pop(offspring1)
            self.insert_classifier_pop(offspring2)

    def get_identical(self, classifier, search_matchset=False):
        if search_matchset:
            identical = [self.popset[ref] for ref in self.matchset if
                         self.cl_methods.is_equal(classifier, self.popset[ref])]
            if identical:
                return identical[0]
        else:
            identical = [cl for cl in self.popset if
                         self.cl_methods.is_equal(classifier, cl)]
            if identical:
                return identical[0]
        return None

# subsumption methods
    def subsume_into_parents(self, offspring, parent1, parent2):
        if self.cl_methods.subsumption(parent1, offspring):
            self.micro_pop_size += 1
            parent1.update_numerosity(1)
        elif self.cl_methods.subsumption(parent2, offspring):
            self.micro_pop_size += 1
            parent2.update_numerosity(1)
        else:
            self.subsume_into_correctset(offspring)

    def subsume_into_correctset(self, classifier):
        choices = [ref for ref in self.correctset if
                   self.cl_methods.subsumption(self.popset[ref], classifier)]
        if choices:
            idx = random.randint(choices.__len__())
            self.popset[choices[idx]].update_numerosity(1)
            self.micro_pop_size += 1
            return
        self.insert_classifier_pop(classifier)

    def subsume_correctset(self):
        subsumer = None
        for ref in self.correctset:
            if self.cl_methods.is_subsumer(self.popset[ref]):
                subsumer = self.popset[ref]
                break
        delete_list = []
        if subsumer:
            delete_list = [ref for ref in self.correctset if
                           self.cl_methods.is_more_general(subsumer, self.popset[ref])]
        for ref in delete_list:
            subsumer.update_numerosity(self.popset[ref].numerosity)
            self.remove_from_pop(ref)
            self.remove_from_matchset(ref)
            self.remove_from_correctset(ref)

# update sets
    def update_sets(self, target):
        m_size = sum([self.popset[ref].numerosity for ref in self.matchset])
        [self.popset[ref].update_params(m_size, target) for ref in self.matchset]
        [self.popset[ref].update_correct() for ref in self.correctset]

    def clear_sets(self):
        self.matchset = []
        self.correctset = []

# evaluation methods
    def pop_average_eval(self):
        generality_sum = sum([(NO_FEATURES - classifier.specified_atts.__len__())/float(NO_FEATURES)
                              for classifier in self.popset])
        loss_sum = sum([classifier.loss for classifier in self.popset])
        try:
            self.ave_generality = generality_sum / float(self.micro_pop_size)
            self.ave_loss = loss_sum / float(self.micro_pop_size)
        except ZeroDivisionError:
            self.ave_generality = None
            self.ave_loss = None

# other methods
    def get_pop_tracking(self):
        tracking = str(self.popset.__len__()) + "\t" + str(self.micro_pop_size) \
                   + "\t" + str("%.4f" % self.ave_loss) + "\t" + str("%.4f" % self.ave_generality)
        return tracking
