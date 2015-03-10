
from math import log
import numpy as np
from LOTlib.Miscellaneous import Infinity, lambdaNone, raise_exception
from LOTlib.GrammarRule import BVUseGrammarRule
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis


class LOTHypothesisVectorized(LOTHypothesis):

    # --------------------------------------------------------------------------------------------------------
    # Compute prior (includes vectorized verion)

    def compute_prior(self):
        """Compute `self.prior` using `self.prior_vector`.

        This is optimized to run as fast as possible:
            - the * operator is the fastest way to do element-wise multiplication between two vectors
            - np.ndarray.sum() is the fastest way to sum a vector

        We compute the prior here by multiplying the grammar rule counts (`self.rules_vector`) element-wise
        with the grammar rule probabilities (`self.grammar_vector`). To get the prior we sum the result.

        """
        if self.rules_vector is None:
            self.set_rules_vector()
        self.set_grammar_vector()

        prior_vector = self.rules_vector * self.grammar_vector
        self.prior = prior_vector.sum() / self.prior_temperature

        self.update_posterior()
        return self.prior

    def set_grammar_vector(self):
        """
        Set `self.grammar_vector` -- this is a vector of rule probabilities:  1  x  [# grammar rules]

        TODO
        ----
        we need to calculate the rule's probability relative to siblings (including BV's)
        --> `Z` & `generation_prob` are inspired by FunctionNode.recompute_generation_probabilities

        """
        Z = {nt: log(sum([r.p for r in self.grammar.rules[nt]])) for nt in self.grammar.nonterminals()}

        def generation_prob(rule):
            return log(rule.p) - Z[rule.nt]

        grammar_vector = np.empty(len(self.rules))
        grammar_vector.fill(-np.inf)
        for r in self.rules:
            grammar_vector[self.rule_idxs[r]] = generation_prob(r)

        self.grammar_vector = grammar_vector
        # self.grammar_vector = [generation_prob(r) for r in self.rules]

    def set_rules_vector(self):
        """
        Compute `self.rules_vector` by collecting counts of each rule used to generate `self.value`.

        This is a vector of rule counts:  1  x  [# grammar rules]

        TODO
        ----
        * BV rules in vector - do we add these as an extra item to count? or what do we do here..?
          > if isinstance(rule, BVUseGrammarRule): ...
        * Should the fix on the line where we set `grammar_rules` be in FunctionNode instead of here?

        Note
        ----
        `rule_indexes` is a hash table of vector indices -- when collecting rule counts this is much
        faster than self.rules.index(rule)  [for grammars with many rules, rules.index() is very expensive]

        """
        self.rules = [r for sublist in self.grammar.rules.values() for r in sublist]
        self.rule_idxs = {r: i for i, r in enumerate(self.rules)}
        self.rules_vector = np.zeros(len(self.rules))

        # Use vector to collect the counts for each GrammarRule used to generate the FunctionNode
        #  `subnodes()` gives a list starting with 2 duplicate nodes, so skip 1st item of list
        grammar_rules = [fn.rule for fn in self.value.subnodes()[1:]]
        for rule in grammar_rules:
            try:
                rule_idx = self.rule_idxs[rule]
                self.rules_vector[rule_idx] += 1
            except Exception:
                if isinstance(rule, BVUseGrammarRule):
                    pass
                else:
                    print Exception
