# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import os.path
from joblib import Parallel, delayed
import random
import time
from collections import Counter

from preprocessing import Preprocessing
from config import *
from reglo_gp import REGLoGP
from visualization import plot_records, plot_bar


def handle_model(args):
    exp, data = args
    model = REGLoGP(exp, data)
    ml_perf, class_prec, track_to_plot = model.train_model()
    return [ml_perf, class_prec, track_to_plot]


def run_parallel(olo, cv, cmplt):
    random.seed(SEED_NUMBER)
    os.makedirs(REPORT_PATH, exist_ok=True)
    os.makedirs(os.path.join(REPORT_PATH, DATA_HEADER), exist_ok=True)

    data = Preprocessing()
    data.main(olo, cv, cmplt)
    if data.data_train_folds:
        n_jobs = data.data_train_folds.__len__()
    else:
        n_jobs = AVG_COUNT

    start = time.time()
    arg_instances = [[idx, data] for idx in range(n_jobs)]
    results = Parallel(n_jobs=n_jobs, verbose=1, backend="multiprocessing")(map(delayed(handle_model), arg_instances))
    end = time.time()
    print('multi-threading time = {:.3f}'.format((end - start)/60))

    ml_performance = [result[0] for result in results]
    class_precision = [result[1] for result in results]
    track_to_plot = [result[2] for result in results]

    avg_perf = avg_performance(ml_performance)
    print('Average ML performance:')
    [print(metric + ' ' + str('%.5f' % val)) for metric, val in avg_perf.items() if metric in ('micro-f', 'macro-f',
                                                                                               'micro-pr', 'macro-pr',
                                                                                               'micro-re', 'macro-re',
                                                                                               'rl')]

    avg_precision = avg_performance(class_precision)
    plot_bar(avg_precision, 'precision')
    plot_records(track_to_plot)


def avg_performance(dict_list):
    total = sum(map(Counter, dict_list), Counter())
    avg_result = {key: total[key] / dict_list.__len__() for key in dict_list[0].keys()}
    return avg_result


if __name__ == "__main__":
    run_parallel(0, 0, 1)
