from LOTHypothesisState import LOTHypothesisState

class LikelihoodLOTHypothesisState(LOTHypothesisState):
    """
    A LOTHypothesisState that uses likelihoods to score rather than posteriors
    """
    def score_terminal_state(self):
        """ Get the score here """
        # We must make this compile the function since it is told not to compile in newh within self.make_children
        self.value.fvalue = self.value.compile_function() # make it actually compile!
        return self.value.compute_likelihood(self.data)
