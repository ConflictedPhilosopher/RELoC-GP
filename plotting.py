# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from config import *


class PlotTrack:
    def __init__(self):
        self.records = np.zeros([int(MAX_ITERATION/TRACK_FREQ), 3])

    def plot_records(self, records):

        for i in range(records.__len__()):
            record = records[i]
            for j in range(int(MAX_ITERATION/TRACK_FREQ)):
                try:
                    self.records[j] += record[j]
                except IndexError:
                    self.records[j] += record[-1]
        self.records /= float(records.__len__())

        iterations = range(0, MAX_ITERATION, TRACK_FREQ)
        # iterations = [int(it) for it in iterations]
        train_loss = self.records[:, 1]
        test_loss = self.records[:, 2]
        plt.plot(iterations, train_loss, label='training loss')
        plt.plot(iterations, test_loss, label='test loss')
        plt.xlabel('Iteration')
        plt.ylabel('Hamming loss')
        plt.legend()
        plt.show()
