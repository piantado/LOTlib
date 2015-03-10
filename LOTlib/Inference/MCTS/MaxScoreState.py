from State import State
from LOTlib.Miscellaneous import Infinity

class MaxScoreState(State):
    """
    A state that scores based on the max we've seen so far down this path, rather than the average
    """

    def get_xbar(self):
        if self.is_complete:
            return -Infinity
        elif self.nsteps == 0: # Just the prior. But it get over-written by Infinity compute_weights
            return None
        else:
            return self.summarystat # since we're going on the max

    def update_score(self, score):
        """ Take a sample from one of my children and update my score. My children call this """
        if score > self.summarystat:
            self.summarystat = score

        self.nsteps   += 1
