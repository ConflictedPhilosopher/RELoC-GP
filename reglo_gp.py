# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------

import os.path
from config import *
from classifier_set import ClassifierSets
from prediction import Prediction
from timer import Timer
from performance import Performance
from reporting import Reporting


class REGLoGP:
    def __init__(self, exp, env):
        self.exp = exp
        self.env = env
        self.tracked_loss = 0
        self.no_match = 0
        self.timer = Timer()

        if REBOOT_MODEL:
            self.population = []
        else:
            self.population = ClassifierSets(env.preprocessing.attribute_info, env.preprocessing.dtypes, self.timer)
            self.iteration = 0
            try:
                track_file = os.path.join(os.path.curdir, REPORT_PATH, "tracking_" + str(self.exp) + ".txt")
                self.training_track = open(track_file, 'w')
            except Exception as exc:
                print(exc)
                print("can not open track_" + str(self.exp) + ".txt")
                raise
            else:
                self.training_track.write("iteration\tmacroPop\tmicroPop\taveLoss"
                                          "\taveGenerality\ttestLoss\ttime(min)\n")
            self.train_model()
            self.timer.get_global_timer()
            print(self.timer.get_timer_report())

    def train_model(self):
        samples_training = self.env.preprocessing.data_train_list
        performance = Performance(self.env.preprocessing.label_count)
        stop_training = False
        loss_old = 1.0
        while self.iteration < MAX_ITERATION and not stop_training:
            sample = samples_training[self.iteration % samples_training.__len__()]
            self.train_iteration(sample)

            def track_performance():
                samples_test = self.env.preprocessing.data_test_list
                loss = 0
                for test_sample in samples_test:
                    self.population.make_eval_matchset(test_sample[0])
                    if not self.population.matchset:
                        self.no_match += 1
                    else:
                        predict = Prediction(self.population.popset, self.population.matchset)
                        label_prediction = predict.max_prediction()
                        loss += performance.hamming_loss(label_prediction, test_sample[1])
                return loss / samples_test.__len__()

            if (self.iteration % TRACK_FREQ) == 0 and self.iteration > 0:
                self.timer.start_evaluation()
                if abs(self.tracked_loss/TRACK_FREQ - loss_old) < ERROR:
                    stop_training = True
                else:
                    loss_old = self.tracked_loss/TRACK_FREQ
                test_loss = track_performance()
                self.population.pop_average_eval()
                self.training_track.write(str(self.iteration) + "\t" + self.population.get_pop_tracking() + "\t"
                                          + str("%.4f" % test_loss) + "\t"
                                          + str("%.4f" % self.timer.get_global_timer()) + "\n")
                self.timer.stop_evaluation()
                self.tracked_loss = 0
            self.iteration += 1
        self.training_track.close()

        self.timer.start_evaluation()
        self.population.pop_average_eval()
        [test_evaluation, test_coverage] = self.evaluation()
        [train_evaluation, train_coverage] = self.evaluation(False)
        self.timer.stop_evaluation()

        reporting = Reporting(self.exp)
        reporting.write_model_stats(self.population.popset, self.timer, train_evaluation, train_coverage,
                                    test_evaluation, test_coverage)
        reporting.write_pop(self.population.popset, self.env.preprocessing.dtypes)

    def train_iteration(self, sample):
        self.population.make_matchset(sample[0], sample[1], self.iteration)
        self.timer.start_evaluation()
        predict = Prediction(self.population.popset, self.population.matchset)
        label_prediction = predict.max_prediction()
        self.tracked_loss += (label_prediction.symmetric_difference(sample[1]).__len__() / self.env.preprocessing.label_count)
        self.timer.stop_evaluation()

        self.population.make_correctset(sample[1])
        self.population.update_sets(sample[1])

        if (self.iteration - self.population.get_time_average()) > THETA_GA:
            [self.population.popset[idx].update_ga_time(self.iteration) for idx in self.population.correctset]
            self.population.apply_ga(self.iteration, sample[0])

        self.population.deletion()
        self.population.clear_sets()

    def evaluation(self, test=True):
        self.no_match = 0
        multi_label_perf = dict()
        performance = Performance(self.env.preprocessing.label_count)
        vote_list = []
        if test:
            samples = self.env.preprocessing.data_test_list
        else:
            samples = self.env.preprocessing.data_train_list

        def update_performance(sample):
            self.population.make_eval_matchset(sample[0])
            label_prediction = {}
            vote = {}

            if not self.population.matchset:
                self.no_match += 1
            else:
                predict = Prediction(self.population.popset, self.population.matchset)
                if PREDICTION_METHOD == 'max':
                    label_prediction = predict.max_prediction()
                else:
                    predict.aggregate_prediction()
                    if THRESHOLD == 'OT':
                        [label_prediction, vote] = predict.one_threshold()
                    elif THRESHOLD == 'RCUT':
                        [label_prediction, vote] = predict.rank_cut()
                    else:
                        print("prediction threshold method unidentified!")
                performance.update_example_based(vote, label_prediction, sample[1])
                performance.update_class_based(label_prediction, sample[1])
            vote_list.append(vote)
            self.population.clear_sets()

        [update_performance(sample) for sample in samples]
        performance.micro_average()
        performance.macro_average()
        """
        target_list = []
        performance.roc(vote_list, target_list)
        """
        multi_label_perf['em'] = performance.exact_match_example / samples.__len__()
        multi_label_perf['hl'] = performance.hamming_loss_example / samples.__len__()
        multi_label_perf['acc'] = performance.accuracy_example / samples.__len__()
        multi_label_perf['pr'] = performance.precision_example / samples.__len__()
        multi_label_perf['re'] = performance.recall_example / samples.__len__()
        multi_label_perf['f'] = performance.fscore_example / samples.__len__()
        multi_label_perf['micro-f'] = performance.micro_fscore
        multi_label_perf['macro-f'] = performance.macro_fscore
        multi_label_perf['micro-pr'] = performance.micro_precision
        multi_label_perf['macro-pr'] = performance.macro_precision
        multi_label_perf['micro-re'] = performance.micro_recall
        multi_label_perf['macro-re'] = performance.macro_recall
        multi_label_perf['1e'] = performance.one_error_example / samples.__len__()
        multi_label_perf['rl'] = performance.rank_loss_example / samples.__len__()
        multi_label_perf['auc'] = performance.roc_auc
        sample_coverage = 1 - (self.no_match / samples.__len__())

        return [multi_label_perf, sample_coverage]