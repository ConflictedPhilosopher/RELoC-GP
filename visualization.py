# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import os.path

import numpy as np
import matplotlib.pyplot as plt

from config import *


def plot_records(records):
    records_plot = np.zeros([int(MAX_ITERATION / TRACK_FREQ), 3])
    for i in range(records.__len__()):
        record = records[i]
        for j in range(int(MAX_ITERATION / TRACK_FREQ)):
            try:
                records_plot[j] += record[j]
            except IndexError:
                records_plot[j] += record[-1]
    records_plot /= float(records.__len__())

    iterations = range(TRACK_FREQ, MAX_ITERATION + TRACK_FREQ, TRACK_FREQ)
    train_f = records_plot[:, 1]
    test_f = records_plot[:, 2]
    plt.plot(iterations, train_f, label='train f-score')
    plt.plot(iterations, test_f, label='test f-score')
    plt.xlabel('Iteration')
    plt.ylabel('F-score')
    plt.legend()
    fig_name = str(MAX_CLASSIFIER) + '-' + str(PROB_HASH) + '.png'
    plt.savefig(os.path.join(os.path.curdir, REPORT_PATH, DATA_HEADER, fig_name), bbox_inches='tight')
    plt.close()


def plot_bar(value_dict, title):
    fig, ax = plt.subplots(figsize=(16, 9))
    xaxis = list(value_dict.values())
    yaxis = list(value_dict.keys())
    ax.barh(yaxis, xaxis)
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)
    ax.set_title('class-specific ' + title, loc='right', )
    plt.show()


def plot_graph(labels, sim_matrix):
    print('label graph...')
    print(labels, sim_matrix)


def plot_image(image_id, labels, prediction):
    print('test image with labels')
    print(image_id, labels, prediction)
