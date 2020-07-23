# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import os.path

import environment as env
from config import *
from reglo_gp import REGLoGP

# parallel run

# average performance

# main



os.makedirs(REPORT_PATH, exist_ok=True)
model_0 = REGLoGP(0, env)
model_0.train_model()

