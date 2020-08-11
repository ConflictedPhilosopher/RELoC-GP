# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------

import random
from copy import deepcopy

from classifier import Classifier
from config import *


class GeneticAlgorithm:
    def __int__(self, attribute_info, dtypes):
        self.attribute_info = attribute_info
        self.dtypes = dtypes
        self.classifier = Classifier()
        self.child1 = self.classifier
        self.child2 = self.classifier

    def selection(self, correct_set, popset, iteration, state):
        parents = []
        if correct_set.__len__() > 1:
            fitness = [popset[i].fitness for i in correct_set]
            if SELECTION == 'r':
                parents.append(deepcopy(popset[self.roulette(fitness, correct_set)]))
                parents.append(deepcopy(popset[self.roulette(fitness, correct_set)]))
            elif SELECTION == 't':
                candidates = popset[correct_set]
                parents.append(deepcopy(self.tournament(candidates)))
                parents.append(deepcopy(self.tournament(candidates)))
            else:
                print("Error: GA selection method not identified.")
                return

            self.child1.classifier_copy(parents[0], iteration)
            self.child2.classifier_copy(parents[1], iteration)

            if random.random() < P_XOVER:
                self.xover()
            self.child1.condition, self.child1.specified_atts = self.mutate(self.child1, state)
            self.child2.condition, self.child2.specified_atts = self.mutate(self.child2, state)

        else:
            parents.append(deepcopy(popset[correct_set[0]]))

    def roulette(self, fitness, correct_set):
        total = float(sum(fitness))
        n = 2
        i = 0
        w, v = fitness[0], correct_set[0]
        while n:
            x = total * (1 - random.random() ** (1.0 / correct_set.__len__()))
            total -= x
            while x > w:
                x -= w
                i += 1
                w, v = fitness[i], correct_set[i]
            w -= x
            yield v
            n -= 1

    def tournament(self, candidates, tsize=5):
        for i in range(candidates.__len__()):
            candidates = random.sample(candidates, tsize)
            yield max(candidates, key=lambda x: x.fitness)

    def xover(self):
        atts_child1 = self.child1.specified_atts
        atts_child2 = self.child2.specified_atts
        cond_child1 = self.child1.condition
        cond_child2 = self.child2.condition
        def swap1(att0):
            cond_child2.append(cond_child1.pop(atts_child1.index(att0)))
            atts_child2.append(att0)
            atts_child1.remove(att0)

        def swap2(att0):
            cond_child1.append(cond_child2.pop(atts_child2.index(att0)))
            atts_child1.append(att0)
            atts_child2.remove(att0)

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
                    cond_child1[idx1] = [min(cond_child1[idx1], cond_child2[idx2]),
                                         max(cond_child1[idx1], cond_child2[idx2])]
                    cond_child2.pop(idx2)
                    atts_child2.remove(att0)
                else:  # absorb ranges into child 2
                    cond_child2[idx2] = [min(cond_child1[idx1], cond_child2[idx2]),
                                         max(cond_child1[idx1], cond_child2[idx2])]
                    cond_child1.pop(idx1)
                    atts_child1.remove(att0)
            else:  # Discrete attribute
                cond_child1[idx1], cond_child2[idx2] = cond_child2[idx2], cond_child1[idx1]

        [swap1(att) for att in set(atts_child1).difference(set(atts_child2)) if random.random() < 0.5]
        [swap2(att) for att in set(atts_child2).difference(set(atts_child1)) if random.random() < 0.5]
        [swap3(att) for att in set(atts_child1).intersection(set(atts_child2)) if random.random() < 0.5]

        self.child1.condition = cond_child1
        self.child1.specified_atts = atts_child1

        self.child2.condition = cond_child2
        self.child2.specified_atts = atts_child2

    def mutate(self, state, child_classifier):
        atts_child = child_classifier.specified_atts
        cond_child = child_classifier.condition

        def mutate_single(idx):
            if idx in atts_child:  # attribute specified in classifier condition
                if random.random() < PROB_HASH:  # remove the specification
                    cond_child.pop(atts_child.index(idx))
                    atts_child.remove(idx)
                elif self.attribute_info[idx]:  # continuous attribute
                    mutate_range = random.random() * float(self.attribute_info[idx][1] - self.attribute_info[idx][0]) / 2
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
                else:
                    pass
            else:  # attribute not specified in classifier condition
                atts_child.append(idx)
                cond_child.append(self.classifier.build_match(state[idx], self.attribute_info[idx, self.dtypes[idx]]))

        mutate_idx = random.random(size=(self.attribute_info.__len__(), 1)) < P_MUT
        [mutate_single(att_idx) for att_idx in enumerate(self.attribute_info.__len__()) if mutate_idx[att_idx]]
        return [cond_child, atts_child]
