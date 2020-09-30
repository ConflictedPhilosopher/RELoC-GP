# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------

import os.path
import random

from config import *
from classifier_set import ClassifierSets
from prediction import Prediction
from timer import Timer
from performance import Performance
from reporting import Reporting
from reboot_model import RebootModel
from visualization import plot_image, plot_graph


class REGLoGP(Prediction):
    def __init__(self, exp, pop_size, prob, data, theta):
        Prediction.__init__(self)
        self.exp = exp
        self.data = data
        self.N = pop_size
        self.p = prob
        self.theta = theta
        self.tracked_loss = 0
        self.no_match = 0
        self.timer = Timer()
        self.track_to_plot = []
        random.seed(SEED_NUMBER + exp)

        if REBOOT_MODEL:
            trained_model = RebootModel(self.exp, self.data.dtypes)
            pop = trained_model.get_model(self.N, self.p)
            self.population = ClassifierSets(data.attribute_info, data.dtypes, random, pop)
            self.population.micro_pop_size = sum([classifier.numerosity for classifier in pop])
            self.population.pop_average_eval()
        else:
            self.population = ClassifierSets(data.attribute_info, data.dtypes, random)

        self.iteration = 1
        try:
            track_file = os.path.join(os.path.curdir, REPORT_PATH, DATA_HEADER, 'params-'+str(self.N)+'-'+str(self.p),
                                      "tracking_" + str(self.exp) + ".csv")
            self.training_track = open(track_file, 'w')
        except Exception as exc:
            print(exc)
            print("can not open track_" + str(self.exp) + ".txt")
            raise
        else:
            self.training_track.write("iteration, macroPop, microPop, aveLoss, "
                                      "aveGenerality, trainFscore, testFscore, time(min)\n")

    def train_model(self):
        if self.data.data_train_folds:
            samples_training = self.data.data_train_folds[self.exp]
            samples_test = self.data.data_valid_folds[self.exp]
        else:
            samples_training = self.data.data_train_list
            samples_test = self.data.data_test_list
        performance = Performance()
        stop_training = False
        loss_old = 1.0
        while self.iteration < (MAX_ITERATION + 1) and not stop_training:
            sample = samples_training[self.iteration % samples_training.__len__()]
            self.train_iteration(sample)

            def track_performance(samples):
                fscore = 0
                label_prediction = set()
                for sample in samples:
                    self.population.make_eval_matchset(sample[0])
                    if not self.population.matchset:
                        fscore += performance.fscore(label_prediction, sample[1])
                    else:
                        if PREDICTION_METHOD == 1:
                            label_prediction = Prediction.max_prediction(self, self.population.popset,
                                                                         self.population.matchset, random.randint)
                        else:
                            if THRESHOLD == 1:
                                label_prediction, _ = Prediction.one_threshold(self, self.population.popset,
                                                                               self.population.matchset, self.theta)
                            elif THRESHOLD == 2:
                                label_prediction, _ = Prediction.rank_cut(self, self.population.popset,
                                                                          self.population.matchset)
                            else:
                                print("prediction threshold method unidentified!")

                        fscore += performance.fscore(label_prediction, sample[1])
                return fscore / samples.__len__()

            if (self.iteration % TRACK_FREQ) == 0 and self.iteration > 0:
                # print('Iteration ', self.iteration)
                self.timer.start_evaluation()
                test_fscore = track_performance(samples_test)
                train_fscore = track_performance(samples_training)
                self.population.pop_average_eval()
                self.training_track.write(str(self.iteration) + ", " + self.population.get_pop_tracking() + ", "
                                          + str("%.4f" % train_fscore) + ", "
                                          + str("%.4f" % test_fscore) + ", "
                                          + str("%.4f" % self.timer.get_global_timer()) + "\n")
                self.timer.stop_evaluation()

                self.track_to_plot.append([self.iteration, train_fscore, test_fscore])

                if float(self.tracked_loss / TRACK_FREQ) - loss_old > 0.1:
                    stop_training = True
                else:
                    loss_old = self.tracked_loss / TRACK_FREQ
                self.tracked_loss = 0
                # self.population.pop_compaction()

            self.iteration += 1

        self.training_track.close()

        self.timer.start_evaluation()
        self.population.pop_average_eval()
        [test_evaluation, test_class_precision, test_coverage] = self.evaluation()
        [train_evaluation, _, train_coverage] = self.evaluation(False)
        self.timer.stop_evaluation()

        reporting = Reporting(self.exp, self.N, self.p)
        reporting.write_pop(self.population.popset, self.data.dtypes)
        reporting.write_model_stats(self.population, self.timer, train_evaluation, train_coverage,
                                    test_evaluation, test_coverage)
        global_time = self.timer.get_global_timer()

        print("Process Time (min): ", round(global_time, 5))

        return [test_evaluation, test_class_precision, self.track_to_plot]

    def train_iteration(self, sample):
        self.timer.start_matching()
        self.population.make_matchset(sample[0], sample[1], self.iteration, self.p)
        self.timer.stop_matching()

        label_prediction = Prediction.max_prediction(self, self.population.popset,
                                                     self.population.matchset, random.randint)
        self.tracked_loss += (label_prediction.symmetric_difference(sample[1]).__len__()
                              / NO_LABELS)

        self.population.make_correctset(sample[1])
        self.population.update_sets(sample[1])

        if DO_SUBSUMPTION:
            self.timer.start_subsumption()
            self.population.subsume_correctset()
            self.timer.stop_subsumption()

        if (self.iteration - self.population.get_time_average()) > THETA_GA:
            popset = self.population.popset
            if self.population.matchset.__len__() > 1:
                self.timer.start_label_partition()
                self.population.apply_partitioning(self.iteration, sample[1])
                self.timer.stop_label_partition()
                # print('target ', [self.data.label_ref[label] for label in sample[1]])
                # cluster_dict = {k: self.population.label_clusters[k] for k in
                #                 range(self.population.label_clusters.__len__())}
                # plot_graph(cluster_dict, self.population.label_similarity, self.data.label_ref)
            if self.population.correctset.__len__() > 0:
                self.timer.start_selection()
                [popset[idx].update_ga_time(self.iteration) for idx in self.population.correctset]
                self.population.apply_ga(self.iteration, sample[0], self.data.data_train_list, self.p)
                self.timer.stop_selection()

        self.timer.start_deletion()
        self.population.deletion(self.N)
        self.timer.stop_deletion()
        self.population.clear_sets()

    def evaluation(self, test=True):
        performance = Performance()
        vote_list = []

        if test:
            if self.data.data_valid_folds:
                samples = self.data.data_valid_folds[self.exp]
            else:
                samples = self.data.data_test_list
        else:
            if self.data.data_train_folds:
                samples = self.data.data_train_folds[self.exp]
            else:
                samples = self.data.data_train_list

        def update_performance(sample):
            self.population.make_eval_matchset(sample[0])
            label_prediction = set()
            vote = {}

            if not self.population.matchset:
                self.no_match += 1
                performance.update_example_based(vote, label_prediction, sample[1])
                performance.update_class_based(label_prediction, sample[1])
            else:
                if PREDICTION_METHOD == 1:
                    label_prediction = Prediction.max_prediction(self, self.population.popset, self.population.matchset,
                                                                 random.randint)
                else:
                    if THRESHOLD == 1:
                        label_prediction, vote = Prediction.one_threshold(self, self.population.popset,
                                                                          self.population.matchset, self.theta)
                    elif THRESHOLD == 2:
                        label_prediction, vote = Prediction.rank_cut(self, self.population.popset,
                                                                     self.population.matchset)
                    else:
                        print("prediction threshold method unidentified!")

                performance.update_example_based(vote, label_prediction, sample[1])
                performance.update_class_based(label_prediction, sample[1])

                if DEMO:
                    self.population.build_graph([self.population.popset[idx] for idx in self.population.matchset])
                    cluster_dict = {0: self.population.predicted_labels}
                    plot_image(sample[2], sample[1], vote, self.data.label_ref)
                    plot_graph(cluster_dict, self.population.label_similarity, self.data.label_ref)

            vote_list.append(vote)
            self.population.clear_sets()

        [update_performance(sample) for sample in samples]
        performance.micro_average()
        performance.macro_average()
        multi_label_perf = performance.get_report(samples.__len__())
        class_precision = {}
        for label in self.data.label_ref.keys():
            class_measure = performance.class_based_measure[label]
            class_precision[self.data.label_ref[label]] = class_measure['TP'] / (
                    class_measure['TP'] + class_measure['FP'] + 1)
        """
        target_list = []
        performance.roc(vote_list, target_list)
        """
        sample_coverage = 1 - (self.no_match / samples.__len__())

        return [multi_label_perf, class_precision, sample_coverage]
