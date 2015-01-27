
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

        for nt in nonterminals:
            # Don't propose to constants!
            if nt.upper() is not 'CONST':
                idxs, r = self.get_rules(rule_nt=nt)
                # Only rules with alternatives/siblings
                if len(idxs) == 1:
                    proposal_indexes.remove(idxs[0])

        return proposal_indexes


class MixtureGrammarHypothesis(GrammarHypothesisVectorized):
    """
    This will let us single out 'MATH' rules & 'INTERVAL' rules as `lambda` & `(1-lambda)`.

    """
    def __init__(self, grammar, hypotheses, value=None, **kwargs):
        if value is None:
            value = np.array([1.0, 1.0])
        GrammarHypothesisVectorized.__init__(self, grammar, hypotheses, value=value, **kwargs)

    def update(self):
        """This is what's different for this class. We only have one value -- lambda in our mixture model.

        We use lambda then to re-set our grammar rules (hand code: MATH == lambda, INTERVAL == (1-lambda).)

        We'll need to do this whenever we calculate things like predictive, because we need to use
          the `posterior_score` of each domain hypothesis get weights for our predictions.

        """
        # Set probability for each rule corresponding to value index
        for rule in self.get_rules(rule_to='MATH')[1]:
            rule.p = self.value[0]
        for rule in self.get_rules(rule_to='INTERVAL')[1]:
            rule.p = self.value[1]

        # Recompute prior for each hypothesis, given new grammar probs
        # for h in self.hypotheses:
        #     h.compute_prior(recompute=True, vectorized=False)
        #     h.update_posterior()

    def get_propose_idxs(self):
        """Get indexes to propose to => for Mix...Hypothesis, we use a special self.value system."""
        return [0, 1]

    def set_value(self, value):
        assert len(value) == 2, "ERROR: Invalid value vector!!!"
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self.value = value
        self.rules = [r for sublist in self.grammar.rules.values() for r in sublist]
        self.update()

    def normalize_value_vector(self):
        """
        here, |x| is 2 instead of |rules|.

        """
        # Make dictionary of normalization constants for each nonterminal
        nt_Z = {}
        for nt in self.grammar.nonterminals():
            Z = sum([r.p for r in self.get_rules(rule_nt=nt)[1]])
            nt_Z[nt] = Z

        # Normalize each probability in `self.value`
        normalized = np.zeros(2)
        normalized[0] = self.value[0] / nt_Z['MATH']
        normalized[1] = self.value[1] / nt_Z['INTERVAL']

        return np.log(normalized)

    def init_C(self):
        """
        Here, C is  `|hypotheses| x 2`  instead of   `|hypotheses| x |rules|`

        """
        self.C = np.zeros((len(self.hypotheses), 2))
        rule_idxs = {'MATH': 0, 'INTERVAL': 1}

        for j, h in enumerate(self.hypotheses):
            grammar_rules = [fn.rule for fn in h.value.subnodes()[1:]]
            for rule in grammar_rules:
                try:
                    if rule.nt in ('MATH', 'INTERVAL'):
                        self.C[j, rule_idxs[rule.nt]] += 1
                except Exception as e:
                    if isinstance(rule, BVUseGrammarRule):
                        pass
                    else:
                        raise e


