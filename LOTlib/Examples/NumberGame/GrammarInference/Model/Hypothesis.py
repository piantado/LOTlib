from LOTlib.Hypotheses.GrammarHypothesisVectorized import GrammarHypothesisVectorized

class NoConstGrammarHypothesis(GrammarHypothesisVectorized):
    """
    Don't propose to rules with 'CONST' as the rhs variable.

    """
    def get_propose_idxs(self):
        proposal_indexes = range(self.n)
        nonterminals = self.grammar.nonterminals()

        # Don't propose to constants
        idxs, r = self.get_rules(rule_nt='CONST')
        for i in idxs:
            proposal_indexes.remove(i)

        # Only rules with alternatives/siblings
        for nt in nonterminals:
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

