
from copy import deepcopy, copy
from math import log
from LOTlib.Miscellaneous import attrmem, logsumexp
from Levenshtein import distance
from LOTlib.Hypotheses.Proposers import ProposalFailedException
from LOTlib.Hypotheses.Likelihoods.StochasticLikelihood import StochasticLikelihood

# from LOTlib.Hypotheses.FactorizedDataHypothesis import FactorizedLambdaHypothesis
# from LOTlib.Hypotheses.FactorizedDataHypothesis import InnerHypothesis


def matching_prefix(x,y):
    for i in xrange(min(len(x), len(y))):
        if x[i] != y[i]:
            return len(y) - i
    return len(y) - min(len(x), len(y))


from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
import random
from LOTlib.Hypotheses.Proposers.InsertDeleteProposal import insert_delete_proposal
from LOTlib.Hypotheses.Proposers.RegenerationProposal import regeneration_proposal
from LOTlib.Hypotheses import Hypothesis
class InnerHypothesis(StochasticLikelihood, LOTHypothesis):
    """
    The type of each function F. This is NOT recursive, but it does allow recurse_ (to refer to the whole lexicon)
    """
    def __init__(self, grammar=None, display="lambda recurse_: %s", **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, display=display, **kwargs)

from LOTlib.Miscellaneous import sample_one
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursionDepthException
from LOTlib.Hypotheses.Lexicon.RecursiveLexicon import RecursiveLexicon
class IncrementalLexiconHypothesis(StochasticLikelihood, RecursiveLexicon):
        """ A hypothesis where we can incrementally add words and propose to only the additions
        """

        def __init__(self, grammar=None, **kwargs):
            RecursiveLexicon.__init__(self, recurse_bound=5, maxnodes=50, variable_weight=3.0, **kwargs)
            self.grammar=grammar
            self.N = 0

        def propose(self):
            while True:
                try:
                    return RecursiveLexicon.propose(self)
                except RecursionDepthException:
                    pass


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

            llcounts = self.make_ll_counts(datum.input, nsamples=512)

            z = log(sum(llcounts.values()))

            # find the largest N such that we get all the strings n<N exactly
            for N in xrange(max([len(r) for r in datum.output.keys()])):
                s1 = {k for k in datum.output.keys() if len(k) <= N }
                s2 = {k for k in llcounts.keys() if len(k) <= N}

                if len(s1 ^ s2) != 0: break # leaving N to be the largest value we found, if the symmetric difference is nonempty
            ll = N*sum(datum.output.values())

            # ll = 0.0  # We are going to compute a pseudo-likelihood, counting close strings as being close
            # for k in datum.output.keys():
            #     # ll += datum.output[k] * (log(llcounts.get(k)) - z) if k in llcounts else -100.0
            #     # ll += datum.output[k] * (log(llcounts.get(k))-z if k in llcounts else -100.0)
            #     # ll += datum.output[k] * (log(llcounts.get(k))-z if k in llcounts else -10000.0)
            #     # ll += datum.output[k] * (log(llcounts.get(k, 1.0e-12)) - z)
            #     # ll += datum.output[k] * logsumexp([ log(llcounts[r])-log(lo) - 1.0 * matching_prefix(r, k) for r in llcounts.keys() ])
            #     # ll += datum.output[k] * logsumexp([ log(llcounts[r])-z - 1.0 * distance(r, k) for r in llcounts.keys() ])
            #     # TODO: Can be sped up by pre-computing the probs once
            #     # ll += datum.output[k] * max([log(llcounts[r]) - z - 1.0 * distance(r, k) for r in llcounts.keys()])
            #
            # # a type prior?
            # # ll = 10000*float(len(set(llcounts.keys()) & set(datum.output.keys()))) / float(len(set(llcounts.keys()) | set(datum.output.keys())))

            return ll


        def propose(self):
            """ We only propose to the last in this kind of lexicon
            """
            new = deepcopy(self)  ## Now we just copy the whole thing
            while True:
                try:
                    i = sample_one(range(self.N))
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
