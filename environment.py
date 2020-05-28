# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------

import os.path
from preprocessing import Preprocessing
from config import *

data_path = os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + ".csv")
train_data_path = os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_train.csv")
test_data_path = os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_test.csv")
fold_path = [os.path.join(DATA_DIR, DATA_HEADER, DATA_HEADER + "_fold_" + str(i + 1) + ".csv") for i in range(5)]

preprocessing = Preprocessing(data_path, None, None)
preprocessing.train_test_split()
preprocessing.characterize_features()
preprocessing.characterize_labels()
preprocessing.multilabel_properties()
preprocessing.data_train_list = preprocessing.format_data(preprocessing.data_train)
preprocessing.data_test_list = preprocessing.format_data(preprocessing.data_test)

