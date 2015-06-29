from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis
from LOTlib.Miscellaneous import log, Infinity, log1mexp, attrmem
from LOTlib.Evaluation.EvaluationException import EvaluationException

# for computing knower-levels
from Data import sample_sets_of_objects, all_objects, word_to_number, ALPHA

# ============================================================================================================
#  Define a class for running MH

GAMMA = -30.0   # the log probability penalty for recursion
LG_1MGAMMA = log1mexp(GAMMA)
MAX_NODES = 50  # How many FunctionNodes are allowed in a hypothesis? If we make this 20, things may slow

from Grammar import grammar
def make_hypothesis(**kwargs):
    """
    Default hypothesis creation
    """
    return NumberExpression(grammar, **kwargs)


class NumberExpression(RecursiveLOTHypothesis):
    
    def __init__(self, grammar=None, value=None, f=None, args=['x'], **kwargs):
        RecursiveLOTHypothesis.__init__(self, grammar, value=value, args=['x'], **kwargs)

    def __call__(self, *args):
        try:
            return RecursiveLOTHypothesis.__call__(self, *args)
        except EvaluationException: # catch recursion and too big
            return None

    @attrmem('prior') # save this in the prior
    def compute_prior(self):
        """Compute the number model prior.

        Log_probability() with a penalty on whether or not recursion is used.

        """
        if self.value.count_nodes() > MAX_NODES:
            return -Infinity
        else:
            if self.value.contains_function(self.recurse):
                recursion_penalty = GAMMA
            else:
                recursion_penalty = LG_1MGAMMA

        return (recursion_penalty + self.grammar.log_probability(self.value)) / self.prior_temperature

    def compute_single_likelihood(self, datum):
        """Computes the likelihood of data.

            TODO: Make sure this precisely matches the number paper.
        """
        response = self(*datum.input)
        if response == 'undef' or response == None:
            return log(1.0/10.0) # if undefined, just sample from a base distribution
        else:
            return log((1.0 - ALPHA)/10.0 + ALPHA * (response == datum.output))

    def get_knower_pattern(self):
        # compute a string describing the behavior of this knower-level
        resp = [ self(set(sample_sets_of_objects(n, all_objects))) for n in xrange(1, 10)]
        return ''.join([str(word_to_number[x]) if (x is not None and x is not 'undef') else 'U' for x in resp])