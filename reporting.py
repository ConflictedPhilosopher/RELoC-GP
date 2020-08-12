# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import os.path
from config import *

from classifier_methods import ClassifierMethods


class Reporting:
    def __init__(self, exp):
        self.exp = exp

    def write_model_stats(self, pop, timer, train_eval, train_coverage, test_eval, test_coverage):
        try:
            file_name = os.path.join(os.path.curdir, REPORT_PATH, DATA_HEADER, "stats_" + str(self.exp) + ".txt")
            stat_file = open(file_name, 'w')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open file', " stats_" + str(self.exp) + ".txt")
            raise

        stat_file.write("Model characterization:\n")
        stat_file.write("Macro pop size: " + str(pop.popset.__len__()) + "\tMicro pop size: " + str(pop.micro_pop_size) +
                        "\tAvg generality:" + str("%.3f" % pop.ave_generality) + "\n\n")

        stat_file.write("Run time (min):\n")
        stat_file.write(timer.get_timer_report())
        stat_file.write("\n")

        stat_file.write("Training instance coverage: " + str("%.4f" % train_coverage) + "\n")
        stat_file.write("Test instance coverage: " + str("%.4f" % test_coverage) + "\n\n")

        stat_file.write("Training performance:\n")
        [stat_file.write(key + "\t") for key in train_eval.keys()]
        stat_file.write('\n')
        [stat_file.write(str("%.4f\t" % value)) for value in train_eval.values()]
        stat_file.write('\n\n')
        stat_file.write("Test performance:\n")
        [stat_file.write(key + "\t") for key in test_eval.keys()]
        stat_file.write('\n')
        [stat_file.write(str("%.4f\t" % value)) for value in test_eval.values()]
        stat_file.close()

    def write_pop(self, pop, dtypes):
        try:
            file_name = os.path.join(os.path.curdir, REPORT_PATH, DATA_HEADER, "model_" + str(self.exp) + ".csv")
            model_file = open(file_name, 'w')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open file', " model_" + str(self.exp) + ".csv")
            raise

        method = ClassifierMethods(dtypes)
        header = ",".join(['f'+str(i) for i in range(NO_FEATURES)])
        header += ", prediction, fitness, hloss, correct_count, numerosity, match_count" \
                  "avg_match_set, init_time, ga_time \n"
        model_file.write(header)
        [model_file.write(method.classifier_print(cl)) for cl in pop]
        model_file.close()
