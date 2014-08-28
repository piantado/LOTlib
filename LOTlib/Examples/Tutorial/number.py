import LOTlib
from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

grammar = Grammar()

grammar.add_rule('START', '', ['EXPR'], 1)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', 'modulo_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', 'exp_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', '', ['NUM'], 10)
grammar.add_rule('NUM', 'one_', None, 0.10)
grammar.add_rule('NUM', 'two_', None, 0.10)
grammar.add_rule('NUM', 'three_', None, 0.10)
grammar.add_rule('NUM', 'four_', None, 0.10)
grammar.add_rule('NUM', 'five_', None, 0.10)
grammar.add_rule('NUM', 'six_', None, 0.10)
grammar.add_rule('NUM', 'seven_', None, 0.10)
grammar.add_rule('NUM', 'eight_', None, 0.10)
grammar.add_rule('NUM', 'nine_', None, 0.10)
grammar.add_rule('NUM', 'ten_', None, 0.10)
grammar.add_rule('NUM', 'n_', None, 4)

for i in range(100):
    print(grammar.generate())


class NumberExpression(LOTHypothesis):

    def __init__(self, grammar, value=None, f=None, proposal_function=None, **kwargs):
        LOTHypothesis.__init__(self, grammar, proposal_function=proposal_function, **kwargs)

        if value is None:
            self.set_value(grammar.generate('EXPR'), f)
        else:
            self.set_value(value, f)

    def copy(self):
        """ Must define this else we return "FunctionHypothesis" as a copy. We need to return a NumberExpression """
        return NumberExpression(grammar, value=self.value.copy(), prior_temperature=self.prior_temperature)

    def compute_prior(self):
        """
                Compute the number model prior
        """
        if self.value.count_nodes() > MAX_NODES:
            self.prior = -Infinity
        else:
            if self.value.contains_function("L_"): recursion_penalty = GAMMA
            else:                                  recursion_penalty = LG_1MGAMMA

            if USE_RR_PRIOR: # compute the prior with either RR or not.
                self.prior = (recursion_penalty + grammar.RR_prior(self.value))  / self.prior_temperature
            else:
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

