
from math import log
import numpy as np
from LOTlib.FunctionNode import FunctionNode
from LOTlib.GrammarRule import BVUseGrammarRule
from LOTlib.Hypotheses.GrammarHypothesisVectorized import GrammarHypothesisVectorized
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis, Infinity
from LOTlib.Examples.NumberGame.NewVersion.Model import NumberGameHypothesis


# ------------------------------------------------------------------------------------------------------------
# NumberGameHypothesis simplified to resemble 'Josh Concepts' model

class JoshConceptsHypothesis(LOTHypothesis):
    """Domain hypothesis class for josh concepts number game with independent rule probs."""
    def __init__(self, grammar, alpha=0.9, domain=100, **kwargs):
        LOTHypothesis.__init__(self, grammar, args=[], **kwargs)
        self.alpha = alpha
        self.domain = domain

    def compute_likelihood(self, data, **kwargs):
        """Likelihood of specified data being produced by this hypothesis.

        Probability is alpha if datum in set, (1-alpha) otherwise).

        """
        s = self()      # set of numbers corresponding to this hypothesis
        error_p = (1.-self.alpha) / self.domain

        def compute_single_likelihood(datum):
            if s is not None and datum in s:
                return log(self.alpha/len(s) + error_p)
            else:
                return log(error_p)

        likelihoods = [compute_single_likelihood(d) for d in data]
        self.likelihood = sum(likelihoods) / self.likelihood_temperature
        self.update_posterior()
        return self.likelihood

class NoDoubleConstNGHypothesis(NumberGameHypothesis):
    def compute_prior(self, recompute=False, vectorized=False):
        """Compute the log of the prior probability.

        """
        # Re-compute the FunctionNode `self.value` generation probabilities
        if recompute:
            self.value.recompute_generation_probabilities(self.grammar)

        # Compute this hypothesis prior
        if self.value.count_subnodes() > self.maxnodes:
            self.prior = -Infinity
        else:
            # Compute prior with either RR or not.
            self.prior = self.value.log_probability() / self.prior_temperature

        # Don't use this tree if we have 2 constants as children in some subnode
        for fn in self.value.subnodes()[1:]:
            if all([arg.name == '' and len(arg.args)==1 and isinstance(arg.args[0], FunctionNode) and arg.args[0].returntype=='OPCONST' for arg in fn.argFunctionNodes()]):
                self.prior = -Infinity
                break

        self.update_posterior()
        return self.prior


# ------------------------------------------------------------------------------------------------------------
# GrammarHypothesis classes

class NoConstGrammarHypothesis(GrammarHypothesisVectorized):
    def get_propose_idxs(self):
        """
        Skip 'Const' rules

        """
        proposal_indexes = range(self.n)
        nonterminals = self.grammar.nonterminals()

        # Don't propose to constants!
        idxs, r = self.get_rules(rule_nt='CONST')
        for i in idxs:
            proposal_indexes.remove(i)

        for nt in nonterminals:
            # Only rules with alternatives/siblings
            idxs, r = self.get_rules(rule_nt=nt)
            if len(idxs) == 1:
                proposal_indexes.remove(idxs[0])

        return proposal_indexes


class MixtureGrammarHypothesis(GrammarHypothesisVectorized):
    """
    This will let us single out 'MATH' rules & 'INTERVAL' rules as `lambda` & `(1-lambda)`.

    """
    def get_propose_idxs(self):
        """This is what's different for this class."""
        return [self.get_rules(rule_to='MATH')[0][0], self.get_rules(rule_to='INTERVAL')[0][0]]

