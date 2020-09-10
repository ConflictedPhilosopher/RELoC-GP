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
from plotting import PlotTrack


def handle_model(args):
    exp, data = args
    model = REGLoGP(exp, data)
    perf, track_to_plot = model.train_model()
    return [perf, track_to_plot]


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
    print('multi-threading time ', (end - start)/60)

    perf = [result[0] for result in results]
    track_to_plot = [result[1] for result in results]
    avg_performance(perf)

    plot = PlotTrack()
    plot.plot_records(track_to_plot)


def avg_performance(perf_dicts):
    total = sum(map(Counter, perf_dicts), Counter())
    avg_perf = {key: val / perf_dicts.__len__() for key, val in total.items()}
    print('Average ML performance:')
    [print(metric + ' ' + str('%.5f' % val)) for metric, val in avg_perf.items()]


if __name__ == "__main__":
    run_parallel(0, 0, 1)
