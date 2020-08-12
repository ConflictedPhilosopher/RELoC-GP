# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import random

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
        self.cl_methods = ClassifierMethods(dtypes)  # Here you can use inheritance instead
        self.classifier = Classifier()  # Here you can use inheritance
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
            new_classifier = Classifier()
            new_classifier.classifier_cover(numerosity_sum + 1, it, state, target,
                                            self.attribute_info, self.dtypes)
            self.insert_classifier_pop(new_classifier, True)
            self.matchset.append(self.popset.__len__() - 1)

    def make_eval_matchset(self, state):

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

# genetic algorithm methods
    def apply_ga(self, iteration, state):
        changed0 = False

        if self.correctset.__len__() > 1:
            parent1, parent2, offspring1, offspring2 = self.selection(iteration)
            if random.random() < P_XOVER and not self.cl_methods.is_equal(offspring1, offspring2):
                offspring1, offspring2, changed0 = self.xover(offspring1, offspring2)
            offspring1.condition, offspring1.specified_atts, changed1 = self.mutate(offspring1, state)
            offspring2.condition, offspring2.specified_atts, changed2 = self.mutate(offspring2, state)
        else:
            parent1 = self.popset[self.correctset[0]]
            parent2 = parent1
            offspring1 = Classifier()
            offspring1.classifier_copy(parent1, iteration)
            offspring2 = Classifier()
            offspring2.classifier_copy(parent2, iteration)

            offspring1.condition, offspring1.specified_atts, changed1 = self.mutate(offspring1, state)
            offspring2.condition, offspring2.specified_atts, changed2 = self.mutate(offspring2, state)

        if changed0:
            offspring1.set_fitness(FITNESS_RED * (offspring1.fitness + offspring2.fitness)/2)
            offspring2.set_fitness(offspring1.fitness)
        else:
            offspring1.set_fitness(FITNESS_RED * offspring1.fitness)
            offspring2.set_fitness(FITNESS_RED * offspring2.fitness)

        if changed0 or changed1 or changed2:
            self.insert_discovered_classifier(offspring1, offspring2, parent1, parent2)

    def selection(self, iteration):
        fitness = [self.popset[i].fitness for i in self.correctset]
        if SELECTION == 'r':
            roulette = self.roulette(fitness)
            parent1 = self.popset[next(roulette)]
            parent2 = self.popset[next(roulette)]
        elif SELECTION == 't':
            candidates = [self.popset[idx] for idx in self.correctset]
            tournament = self.tournament(candidates)
            parent1 = next(tournament)
            parent2 = next(tournament)
        else:
            print("Error: GA selection method not identified.")
            return

        offspring1 = Classifier()
        offspring1.classifier_copy(parent1, iteration)
        offspring2 = Classifier()
        offspring2.classifier_copy(parent2, iteration)

        return [parent1, parent2, offspring1, offspring2]

    def roulette(self, fitness):
        total = float(sum(fitness))
        n = 2
        i = 0
        w, v = fitness[0], self.correctset[0]
        while n:
            x = total * (1 - random.random() ** (1.0 / self.correctset.__len__()))
            total -= x
            while x > w:
                x -= w
                i += 1
                w, v = fitness[i], self.correctset[i]
            w -= x
            yield int(v)
            n -= 1

    def tournament(self, candidates, tsize=5):
        for i in range(candidates.__len__()):
            candidates = random.sample(candidates, min(candidates.__len__(), tsize))
            yield max(candidates, key=lambda x: x.fitness)

    def xover(self, offspring1, offspring2):
        changed = False
        atts_child1 = offspring1.specified_atts
        atts_child2 = offspring2.specified_atts
        cond_child1 = offspring1.condition
        cond_child2 = offspring2.condition

        def swap1(att0):
            cond_child2.append(cond_child1.pop(atts_child1.index(att0)))
            atts_child2.append(att0)
            atts_child1.remove(att0)
            return True

        def swap2(att0):
            cond_child1.append(cond_child2.pop(atts_child2.index(att0)))
            atts_child1.append(att0)
            atts_child2.remove(att0)
            return True

        def swap3(att0):
            idx1 = atts_child1.index(att0)
            idx2 = atts_child2.index(att0)
            if self.dtypes[att0]:  # Continuous attribute
                choice_key = random.randint(0, 3)
                if choice_key == 0:  # swap min of the range
                    cond_child1[idx1][0], cond_child2[idx2][0] = cond_child2[idx2][0], cond_child1[idx1][0]
                elif choice_key == 1:  # swap max of the range
                    cond_child1[idx1][1], cond_child2[idx2][1] = cond_child2[idx2][1], cond_child1[idx1][1]
                elif choice_key == 2:  # absorb ranges into child 1
                    cond_child1[idx1] = [min(cond_child1[idx1][0], cond_child2[idx2][0]),
                                         max(cond_child1[idx1][1], cond_child2[idx2][1])]
                    cond_child2.pop(idx2)
                    atts_child2.remove(att0)
                else:  # absorb ranges into child 2
                    cond_child2[idx2] = [min(cond_child1[idx1][0], cond_child2[idx2][0]),
                                         max(cond_child1[idx1][1], cond_child2[idx2][1])]
                    cond_child1.pop(idx1)
                    atts_child1.remove(att0)
            else:  # Discrete attribute
                cond_child1[idx1], cond_child2[idx2] = cond_child2[idx2], cond_child1[idx1]
            return True

        changed = [swap1(att) for att in set(atts_child1).difference(set(atts_child2)) if random.random() < 0.5]
        changed = [swap2(att) for att in set(atts_child2).difference(set(atts_child1)) if random.random() < 0.5]
        changed = [swap3(att) for att in set(atts_child1).intersection(set(atts_child2)) if random.random() < 0.5]

        offspring1.condition = cond_child1
        offspring1.specified_atts = atts_child1
        offspring2.condition = cond_child2
        offspring2.specified_atts = atts_child2

        return [offspring1, offspring2, changed]

    def mutate(self, child_classifier, state):
        changed = False
        atts_child = child_classifier.specified_atts
        cond_child = child_classifier.condition

        def mutate_single(idx):
            if idx in atts_child:  # attribute specified in classifier condition
                if random.random() < PROB_HASH:  # remove the specification
                    ref_2_cond = atts_child.index(idx)
                    atts_child.remove(idx)
                    cond_child.pop(ref_2_cond)
                    return True
                elif self.dtypes[idx]:  # continuous attribute
                    mutate_range = random.random() * float(self.attribute_info[idx][1] -
                                                           self.attribute_info[idx][0]) / 2
                    if random.random() < 0.5:  # mutate min of the range
                        if random.random() < 0.5:  # add
                            cond_child[atts_child.index(idx)][0] += mutate_range
                        else:  # subtract
                            cond_child[atts_child.index(idx)][0] -= mutate_range
                    else:  # mutate max of the range
                        if random.random() < 0.5:  # add
                            cond_child[atts_child.index(idx)][1] += mutate_range
                        else:  # subtract
                            cond_child[atts_child.index(idx)][1] -= mutate_range
                    cond_child[atts_child.index(idx)].sort()
                    return True
                else:
                    pass
            else:  # attribute not specified in classifier condition
                atts_child.append(idx)
                cond_child.append(self.classifier.build_match(state[idx], self.attribute_info[idx], self.dtypes[idx]))
                return True

        changed = [mutate_single(att_idx) for att_idx in range(self.attribute_info.__len__()) if random.random() < P_MUT]
        return [cond_child, atts_child, changed]

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
            if offspring1.specified_atts.__len__() > 0:
                self.subsume_into_parents(offspring1, parent1, parent2)
            if offspring2.specified_atts.__len__() > 0:
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

    def get_time_average(self):
        numerosity_sum = sum([self.popset[idx].numerosity for idx in self.correctset])
        time_sum = sum([(self.popset[idx].ga_time * self.popset[idx].numerosity) for idx in self.correctset])
        return time_sum / float(numerosity_sum)

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
        tracking = str(self.popset.__len__()) + ", " + str(self.micro_pop_size) \
                   + ", " + str("%.4f" % self.ave_loss) + ", " + str("%.4f" % self.ave_generality)
        return tracking
