# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
from time import time


class Timer:
    def __init__(self):
        self.global_start_ref = time()
        self.global_time = 0.0
        self.added_time = 0.0     # time to reboot a model

        # Matching Time Variable
        self.start_ref_matching = 0.0
        self.global_matching = 0.0

        # Label partitioning Time Variable
        self.start_ref_label_partition = 0.0
        self.global_label_partition = 0.0

        # Deletion Time Variables
        self.start_ref_deletion = 0.0
        self.global_deletion = 0.0

        # Subsumption Time Variables
        self.start_ref_subsumption = 0.0
        self.global_subsumption = 0.0

        # Selection Time Variables
        self.start_ref_selection = 0.0
        self.global_selection = 0.0

        # Evaluation Time Variables
        self.start_ref_evaluation = 0.0
        self.global_evaluation = 0.0

        # ************************************************************

    def start_label_partition(self):
        """ Tracks Label Partitioning Time """
        self.start_ref_label_partition = time()

    def stop_label_partition(self):
        """ Tracks Label Partitioning Time """
        diff = time() - self.start_ref_label_partition
        self.global_label_partition += diff

    def start_matching(self):
        """ Tracks MatchSet Time """
        self.start_ref_matching = time()

    def stop_matching(self):
        """ Tracks MatchSet Time """
        diff = time() - self.start_ref_matching
        self.global_matching += diff

    def start_deletion(self):
        """ Tracks Deletion Time """
        self.start_ref_deletion = time()

    def stop_deletion(self):
        """ Tracks Deletion Time """
        diff = time() - self.start_ref_deletion
        self.global_deletion += diff

    def start_subsumption(self):
        """Tracks Subsumption Time """
        self.start_ref_subsumption = time()

    def stop_subsumption(self):
        """Tracks Subsumption Time """
        diff = time() - self.start_ref_subsumption
        self.global_subsumption += diff

    def start_selection(self):
        """ Tracks Selection Time """
        self.start_ref_selection = time()

    def stop_selection(self):
        """ Tracks Selection Time """
        diff = time() - self.start_ref_selection
        self.global_selection += diff

    def start_evaluation(self):
        """ Tracks Evaluation Time """
        self.start_ref_evaluation = time()

    def stop_evaluation(self):
        """ Tracks Evaluation Time """
        diff = time() - self.start_ref_evaluation
        self.global_evaluation += diff

    def get_global_timer(self):
        """ Set the global end timer, call at the very end of algorithm. """
        self.global_time = (time() - self.global_start_ref) + self.added_time
        return self.global_time / 60.0

    def get_timer_report(self):
        """ Reports the time summaries for this run. Returns a string ready to be printed out."""
        output_time = "Global Time\t" + str(self.global_time / 60.0) + \
                      "\nMatching Time\t" + str(self.global_matching / 60.0) + \
                      "\nPartitioning Time\t" + str(self.global_label_partition / 60.0) + \
                      "\nDeletion Time\t" + str(self.global_deletion / 60.0) + \
                      "\nSubsumption Time\t" + str(self.global_subsumption / 60.0) + \
                      "\nSelection Time\t" + str(self.global_selection / 60.0) + \
                      "\nEvaluation Time\t" + str(self.global_evaluation / 60.0) + "\n"

        return output_time

