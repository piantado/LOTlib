from LOTlib.Hypotheses.GrammarHypothesisVectorized import GrammarHypothesisVectorized
from LOTlib.Miscellaneous import logsumexp, exp

class NumberGameGrammarHypothesis(GrammarHypothesisVectorized):

    # --------------------------------------------------------------------------------------------------------
    # p (y in C | H)  where H is our hypothesis space
    #
    # Returns:
    #   A dict of probabilities, values store the probability associated with generating each key.
    #

    def in_concept_mle(self, domain):
        """
        p(y in C | h_mle)   where h_mle is the max likelihood hypothesis

        """
        h_mle = self.get_top_hypotheses(n=1, key=(lambda x: x.likelihood))[0]
        C = h_mle()
        likelihoods = {}
        for y in domain:
            if y in C:
                likelihoods[y] = 1.
            else:
                likelihoods[y] = (h_mle.alpha - 1)
        return likelihoods

    def in_concept_map(self, domain):
        """
        p(y in C | h_map)   where h_map is the max a posteriori hypothesis

        """
        h_map = self.get_top_hypotheses(n=1, key=(lambda x: x.posterior_score))[0]
        C = h_map()
        likelihoods = {}
        for y in domain:
            if y in C:
                likelihoods[y] = 1.
            else:
                likelihoods[y] = (h_map.alpha - 1)
        return likelihoods

    # TODO is this right ?????????????
    def in_concept_avg(self, domain):
        """
        p(y in C | `self.hypotheses`)

        for each hypothesis h, if y in C_h, accumulated w_h where w is the weight of a hypothesis,
        determined by the hypothesis's posterior score p(h | y)

        ==> This is the weighted bayesian model averaging described in (Murphy, 2007)

        """
        self.update()
        probs_in_c = {}

        for y in domain:
            prob_in_c = 0
            Z = logsumexp([h.posterior_score for h in self.hypotheses])

            # for h in self.hypotheses:
            #     h.set_value(h.value)
            # print self.hypotheses[0].prior, self.hypotheses[3].prior, self.hypotheses[5].prior

            for h in self.hypotheses:
                C = h()
                w = h.posterior_score - Z
                if y in C:
                    prob_in_c += exp(w)
            probs_in_c[y] = prob_in_c

        return probs_in_c


class NoConstGrammarHypothesis(GrammarHypothesisVectorized):
    """
    Don't propose to rules with 'CONST' as the rhs variable.

    """

    def get_propose_mask(self):
        """Only propose to rules with other rules with same NT."""
        propose_mask = [True] * self.n

        # Don't propose to constants
        idxs, r = self.get_rules(rule_nt='CONST')
        for i in idxs:
            propose_mask[i] = False

        # Only rules with alternatives/siblings
        for i, nt in enumerate(self.grammar.nonterminals()):
            idxs, r = self.get_rules(rule_nt=nt)
            if len(idxs) == 1:
                propose_mask[i] = False

        return propose_mask


class MixtureGrammarHypothesis(GrammarHypothesisVectorized):
    """
    This will let us single out 'MATH' rules & 'INTERVAL' rules as `lambda` & `(1-lambda)`.

    """

    def get_propose_mask(self):
        """Only propose to rules with other rules with same NT."""
        propose_mask = [False] * self.n

        propose_mask[self.get_rules(rule_to='MATH')[0][0]] = True
        propose_mask[self.get_rules(rule_to='INTERVAL')[0][0]] = True

