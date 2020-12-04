# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------

from os.path import join, curdir
import random

from classifier_set import ClassifierSets
from prediction import *
from timer import Timer
from performance import Performance, fscore
from reporting import Reporting
from reboot_model import RebootModel
from visualization import plot_image, plot_graph
from analyze_model import analyze


class REGLoGP:
    def __init__(self, exp, data):
        self.exp = exp
        self.data = data
        self.tracked_loss = 0
        self.no_match = 0
        self.timer = Timer()
        self.track_to_plot = []
        random.seed(SEED_NUMBER + exp)
        """
        sim_mode = {global, local}
        sim_delta = (0, 1]
        clustering_mode = {None, hfps, wsc}
        """
        if REBOOT_MODEL:
            trained_model = RebootModel(self.exp, self.data.dtypes)
            pop = trained_model.get_model()
            self.population = ClassifierSets(attribute_info=data.attribute_info, dtypes=data.dtypes, rand_func=random,
                                             sim_mode='global', sim_delta=0.1, clustering_method=None,
                                             cosine_matrix=self.data.sim_matrix, popset=pop)
            self.population.micro_pop_size = sum([classifier.numerosity for classifier in pop])
            self.population.pop_average_eval()
            analyze(pop, data)
        else:
            self.population = ClassifierSets(attribute_info=data.attribute_info, dtypes=data.dtypes, rand_func=random,
                                             sim_mode='global', sim_delta=0.9, clustering_method=None,
                                             cosine_matrix=self.data.sim_matrix, data_cov_inv=self.data.cov_inv)

        self.iteration = 1
        try:
            track_file = join(curdir, REPORT_PATH, DATA_HEADER, "tracking_" + str(self.exp) + ".csv")
            self.training_track = open(track_file, 'w')
        except Exception as exc:
            print(exc)
            print("can not open track_" + str(self.exp) + ".txt")
            raise
        else:
            self.training_track.write("iteration, macroPop, microPop, aveFitness, "
                                      "aveGenerality, trainFscore, testFscore, time(min)\n")

    def train_model(self):
        if self.data.data_train_folds:
            samples_training = self.data.data_train_folds[self.exp]
            samples_test = self.data.data_valid_folds[self.exp]
        else:
            samples_training = self.data.data_train_list
            samples_test = self.data.data_test_list
        stop_training = False
        loss_old = 1.0

        if THRESHOLD == 1:
            bi_partition = one_threshold
        elif THRESHOLD == 2:
            bi_partition = rank_cut
        else:
            raise Exception("prediction threshold method unidentified!")

        def track_performance(samples):
            f_score = 0
            label_prediction = set()
            for sample in samples:
                self.population.make_eval_matchset(sample[0])
                if not self.population.matchset:
                    f_score += fscore(label_prediction, sample[1])
                else:
                    if PREDICTION_METHOD == 1:
                        label_prediction = max_prediction([self.population.popset[ref] for ref in
                                                                            self.population.matchset], random.randint)
                    else:
                        vote = aggregate_prediction([self.population.popset[ref]
                                                     for ref in self.population.matchset])
                        label_prediction = bi_partition(vote)
                    f_score += fscore(label_prediction, sample[1])
            return f_score / samples.__len__()

        while self.iteration < (MAX_ITERATION + 1) and not stop_training:
            sample = samples_training[self.iteration % samples_training.__len__()]
            self.train_iteration(sample)

            if (self.iteration % TRACK_FREQ) == 0 and self.iteration > 0:
                self.timer.start_evaluation()
                test_fscore = track_performance(samples_test)
                train_fscore = track_performance(samples_training)
                self.population.pop_average_eval()
                self.training_track.write(str(self.iteration) + ", " + self.population.get_pop_tracking() + ", "
                                          + str("%.4f" % train_fscore) + ", "
                                          + str("%.4f" % test_fscore) + ", "
                                          + str("%.4f" % self.timer.get_global_timer()) + "\n")
                self.timer.stop_evaluation()

                self.track_to_plot.append([self.iteration, train_fscore, test_fscore, self.population.ave_fitness,
                                           float(self.population.micro_pop_size/MAX_CLASSIFIER),
                                           float(self.population.popset.__len__()/MAX_CLASSIFIER)])

                # if float(self.tracked_loss / TRACK_FREQ) - loss_old > 0.1:
                #     stop_training = True
                # else:
                # loss_old = self.tracked_loss / TRACK_FREQ
                self.tracked_loss = 0

            self.iteration += 1

        self.training_track.close()

        self.timer.start_evaluation()
        self.population.pop_average_eval()
        self.population.estimate_label_pr(samples_training)
        [test_evaluation, test_class_precision, test_coverage] = self.evaluation()
        [train_evaluation, _, train_coverage] = self.evaluation(False)
        self.timer.stop_evaluation()

        reporting = Reporting(self.exp)
        reporting.write_pop(self.population.popset, self.data.dtypes)
        global_time = self.timer.get_global_timer()
        reporting.write_model_stats(self.population, self.timer, train_evaluation, train_coverage,
                                    test_evaluation, test_coverage)

        print("Process Time (min): ", round(global_time, 5))

        return [test_evaluation, test_class_precision, self.track_to_plot]

    def train_iteration(self, sample):
        self.timer.start_matching()
        label_prediction = self.population.make_matchset(sample[0], sample[1], self.iteration)
        self.timer.stop_matching()
        self.tracked_loss += (label_prediction.symmetric_difference(sample[1]).__len__()
                              / NO_LABELS)
        self.population.update_sets(sample[1])

        self.population.make_correctset(sample[1])
        if DO_SUBSUMPTION:
            self.timer.start_subsumption()
            self.population.subsume_correctset()
            self.timer.stop_subsumption()

        if self.population.correctset and (self.iteration - self.population.get_time_average()) > THETA_GA:
            popset = self.population.popset
            if self.population.correctset.__len__() > 0:
                self.timer.start_selection()
                [popset[idx].update_ga_time(self.iteration) for idx in self.population.correctset]
                self.population.apply_ga(self.iteration, sample[0], self.data.data_train_list)
                self.timer.stop_selection()

        self.timer.start_deletion()
        self.population.deletion()
        self.timer.stop_deletion()
        self.population.clear_sets()

    def evaluation(self, test=True):
        performance = Performance()
        vote_list = []
        self.no_match = 0

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
        if THRESHOLD == 1:
            bi_partition = one_threshold
        elif THRESHOLD == 2:
            bi_partition = rank_cut
        else:
            raise Exception("prediction threshold method unidentified!")

        def get_prediction_prob(sample):
            self.population.make_eval_matchset(sample[0])
            vote0 = {}
            if not self.population.matchset:
                self.no_match += 1
            else:
                if PREDICTION_METHOD == 1:
                    # TODO max prediction not consistent with the remainder
                    label_prediction = max_prediction([self.population.popset[ref] for ref in
                                                       self.population.matchset], random.randint)
                else:
                    vote0 = aggregate_prediction([self.population.popset[ref] for ref
                                                  in self.population.matchset])
                if DEMO:
                    self.population.build_sim_graph([self.population.popset[idx] for idx in self.population.matchset])
                    cluster_dict = {0: self.population.predicted_labels}
                    plot_image(sample[2], sample[1], vote0, self.data.label_ref)
                    plot_graph(cluster_dict, self.population.label_similarity, self.data.label_ref)

                    for idx in self.population.matchset:
                        if self.population.popset[idx].match_count > 0:
                            print('Classifier acc:')
                            for k, v in self.population.popset[idx].label_based.items():
                                print(self.data.label_ref[k], round(v, 3))

            vote_list.append(vote0)
            self.population.clear_sets()

        [get_prediction_prob(sample) for sample in samples]
        target_list = [sample[1] for sample in samples]
        theta = optimize_theta(vote_list, target_list)

        for t, vote in zip(target_list, vote_list):
            prediction = bi_partition(vote, theta)
            performance.update_example_based(vote, prediction, t)
            performance.update_class_based(prediction, t)

        performance.micro_average()
        performance.macro_average()
        performance.roc(vote_list, target_list)
        multi_label_perf = performance.get_report(samples.__len__())

        class_precision = {}
        for label in self.data.label_ref.keys():
            class_measure = performance.class_based_measure[label]
            class_precision[self.data.label_ref[label]] = class_measure['TP'] / (
                    class_measure['TP'] + class_measure['FP'] + 1)
        sample_coverage = 1 - (self.no_match / samples.__len__())

        return [multi_label_perf, class_precision, sample_coverage]
