
from copy import deepcopy, copy
from math import log
from LOTlib.Miscellaneous import attrmem, logsumexp, sample_one
from Levenshtein import distance
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from LOTlib.Hypotheses.Likelihoods.StochasticLikelihood import StochasticLikelihood
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.Proposers import IDR_proposal

class InnerHypothesis(StochasticLikelihood, LOTHypothesis):
    """
    The type of each function F. This is NOT recursive, but it does allow recurse_ (to refer to the whole lexicon)
    """
    def __init__(self, grammar=None, display="lambda recurse_: %s", **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, display=display, **kwargs)


    def propose(self, **kwargs):
        ret_value, fb = None, None
        while True: # keep trying to propose
            try:
                ret_value, fb = IDR_proposal(self.grammar, self.value, **kwargs)
                break
            except ProposalFailedException:
                pass

        ret = self.__copy__(value=ret_value)

        return ret, fb


from LOTlib.Hypotheses.Lexicon.RecursiveLexicon import RecursiveLexicon
class IncrementalLexiconHypothesis(StochasticLikelihood, RecursiveLexicon):
        """ A hypothesis where we can incrementally add words and propose to only the additions
        """

        def __init__(self, grammar=None, **kwargs):
            RecursiveLexicon.__init__(self, recurse_bound=5, maxnodes=50, variable_weight=3.0, **kwargs)
            self.grammar=grammar
            self.N = 0

        def make_hypothesis(self, **kwargs):
            return InnerHypothesis(**kwargs)

        @attrmem('likelihood')
        def compute_likelihood(self, data, shortcut=None):
            # We'll just be overwriting this
            assert len(data) == 1
            return self.compute_single_likelihood(data[0])

        def compute_single_likelihood(self, datum):
            """
                Compute the likelihood with a Levenshtein noise model
            """

            assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"

            llcounts = self.make_ll_counts(datum.input, nsamples=5000) # Increased nsamples to get the tail of the distribution better

            z = log(sum(llcounts.values()))

            # real likelihood, with a little smoothing
            ll = 0.0  # We are going to compute a pseudo-likelihood, counting close strings as being close
            for k in datum.output.keys():
                ll += datum.output[k] * ((log(llcounts.get(k)) - z) if k in llcounts else -100.0)

            return ll


        def propose(self):

            new = deepcopy(self)  ## Now we just copy the whole thing
            while True:
                try:
                    i = sample_one(range(self.N)) # random one
                    # i = max(self.value.keys()) # only propose to last
                    x, fb = self.get_word(i).propose()
                    new.set_word(i, x)

                    new.grammar = self.value[0].grammar # keep the grammar the same object

                    return new, fb

                except ProposalFailedException:
                    pass

        def dispatch_word(self, word):
            """ We override this so that the hypothesis defaultly calls with the last word via __call__"""
            self.recursive_call_depth += 1

            if self.recursive_call_depth > self.recursive_depth_bound:
                return ''
            else:
                return self.value[word](self.dispatch_word)  # pass in "self" as lex, using the recursive version

        def recursive_call(self, word, *args):
            raise NotImplementedError

        def __call__(self, *args):
            """
            Wrap in self as a first argument that we don't have to in the grammar. This way, we can use self(word, X Y) as above.
            """
            assert self.N > 0, "Cannot call unless something is defined!"

            self.recursive_call_depth = 0

            return self.dispatch_word(self.N-1) # run the last word on calls
