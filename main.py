# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import os.path
from joblib import Parallel, delayed
import random
import multiprocessing
import time
from collections import Counter

from preprocessing import Preprocessing
from config import *
from reglo_gp import REGLoGP


# parallel run
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

    def handle_model(exp, data_p, return_dict_p):
        # exp, data, return_dict = args
        model = REGLoGP(exp, data_p)
        perf = model.train_model()
        return_dict_p[exp] = perf
        return perf

    # start = time.time()
    # arg_instances = [[idx, data, dict()] for idx in range(n_jobs)]
    # results = Parallel(n_jobs=n_jobs, verbose=1, backend="multiprocessing")(map(delayed(handle_model), arg_instances))
    # end = time.time()
    # print('multi-threading time ', (end - start)/60)
    # avg_performance(results)

    start = time.time()
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(n_jobs):
        p = multiprocessing.Process(target=handle_model, args=(i, data, return_dict))
        jobs.append(p)
        p.start()
    for process in jobs:
        process.join()
    end = time.time()
    print('multi-processing time ', (end - start)/60)
    avg_performance(return_dict.values())


def avg_performance(perf_dicts):
    total = sum(map(Counter, perf_dicts), Counter())
    avg_perf = {key: val / perf_dicts.__len__() for key, val in total.items()}
    print('Average ML performance:')
    [print(metric + ' ' + str('%.5f' % val)) for metric, val in avg_perf.items()]


if __name__ == "__main__":
    run_parallel(0, 1, 0)