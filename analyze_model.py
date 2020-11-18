# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np

from config import *


def analyze(pop, data):
    pop = [rule for rule in pop if rule.match_count > 0]

    column_names = list(np.arange(NO_FEATURES)) + ['feature_count', 'labels', 'parent_labels', 'precisions', 'fitness',
                                                   'loss', 'num', 'm_count', 'avg_m_set', 'init_time']
    model = pd.DataFrame(columns=column_names, dtype=None, index=list(np.arange(pop.__len__())))
    for row, rule in enumerate(pop):
        condition = rule.condition
        att_idx = rule.specified_atts
        for c, i in zip(condition, att_idx):
            model.iat[row, i] = c
        model.at[row, 'feature_count'] = att_idx.__len__()
    model['labels'] = [set(classifier.label_based.keys()) for classifier in pop]
    model['parent_labels'] = [classifier.parent_prediction for classifier in pop]
    model['precisions'] = [classifier.label_based for classifier in pop]
    model['fitness'] = [classifier.fitness for classifier in pop]
    model['loss'] = [classifier.loss for classifier in pop]
    model['num'] = [classifier.numerosity for classifier in pop]
    model['m_count'] = [classifier.match_count for classifier in pop]
    model['avg_m_set'] = [classifier.ave_matchset_size for classifier in pop]
    model['init_time'] = [classifier.init_time for classifier in pop]
    learned_labels = set.union(*model['labels'])
    missed_labels = set(np.arange(NO_LABELS)).difference(learned_labels)
    if missed_labels:
        print('missed classes {}'.format(*[data.label_ref[l] for l in missed_labels]))

    # Label space analysis
    prediction = pd.DataFrame(index=list(data.label_ref.values()), columns=['sample_ratio', 'rule_ratio', 'avg_precision'])
    prediction.fillna(0, inplace=True)
    for labelset, num in zip(model['labels'], model['num']):
        for l in labelset:
            prediction.at[data.label_ref[l], 'rule_ratio'] += 1
    for precision, num in zip(model['precisions'], model['num']):
        for l in precision.keys():
            prediction.at[data.label_ref[l], 'avg_precision'] += precision[l]
    prediction['avg_precision'] /= prediction['rule_ratio']
    prediction['rule_ratio'] /= pop.__len__()
    prediction['sample_ratio'] = data.class_ratio.values()
