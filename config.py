# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------

# paths
# file headers
# training parameters (primary, secondary)
# evaluation parameters

SEED_NUMBER = 161

DATA_DIR = "D:\Datasets"
DATA_HEADER = "emotions"
REPORT_PATH = "report"
NO_FEATURES = 72

REBOOT_MODEL = False
MAX_ITERATION = 3000
MAX_CLASSIFIER = 500
PROB_HASH = 0.5
DO_SUBSUMPTION = True

PREDICTION_METHOD = 2  # 1: max prediction - 2: aggregated prediction
THRESHOLD = 'OT'
THETA = 0.5
RANK_CUT = 1

# ------------------------------------------------------------------------------
INIT_FITNESS = 0.01
FITNESS_RED = 0.1
NU = 1
THETA_SUB = 200
LOSS_SUB = 0.01
BETA = 0.1
TRACK_FREQ = 500
ERROR = 1e-3
DELTA = 0.1
THETA_DEL = 20

# GA parameters
SELECTION = 't'   # 'r': roulette wheel selection - 't': tournament selection
P_XOVER = 0.8
P_MUT = 0.04
THETA_GA = 50

