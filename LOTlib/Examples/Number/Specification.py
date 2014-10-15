from LOTlib.Hypotheses import LOTHypothesis
from LOTlib.Miscellaneous import log, Infinity
from Global import ALPHA, GAMMA, LG_1MGAMMA, MAX_NODES

##########################################################
# Define a class for running MH

class NumberExpression(LOTHypothesis):
    
    def __init__(self, grammar, value=None, f=None, proposal_function=None, **kwargs):
        LOTHypothesis.__init__(self, grammar, value=value, proposal_function=proposal_function, **kwargs)

    def compute_prior(self):
        """
            Compute the number model prior: log_probability() with a penalty on whether or not recursion is used
        """
        recursion_penalty = 0
        if self.value.count_nodes() > MAX_NODES:
            self.prior = -Infinity
        else:
            if self.value.contains_function("L_"): 
                recursion_penalty = GAMMA
            else:
                recursion_penalty = LG_1MGAMMA

        self.prior = (recursion_penalty + self.value.log_probability())  / self.prior_temperature
        self.posterior_score = self.prior + self.likelihood

        return self.prior

    def compute_single_likelihood(self, datum):
        """
            Computes the likelihood of data.
            TODO: Make sure this precisely matches the number paper.
        """
        response = self(*datum.input)
        if response == 'undef' or response == None:
            return log(1.0/10.0) # if undefined, just sample from a base distribution
        else:
            return log( (1.0 - ALPHA)/10.0 + ALPHA * ( response == datum.output ) )

