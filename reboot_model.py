# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import os.path

import pandas as pd

from classifier import Classifier
from config import *


class RebootModel():
    def __init__(self, exp, dtypes):
        self.exp = exp
        self.dtypes = dtypes

    def get_model(self):
        try:
            file_name = os.path.join(os.path.curdir, REPORT_PATH, DATA_HEADER, "model_" + str(self.exp) + ".csv")
            model = pd.read_csv(file_name)
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open file', " model_" + str(self.exp) + ".csv")
            raise

        pop = [self.build_classifier(info) for _, info in model.iterrows()]
        return pop

    def build_classifier(self, info):
        new_classifier = Classifier()
        new_classifier.classifier_reboot(info, self.dtypes)
        return new_classifier


if __name__ == "__main__":
    reboot = RebootModel(0, [])
    reboot.get_model()
