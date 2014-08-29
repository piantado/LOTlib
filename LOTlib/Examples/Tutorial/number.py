
from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Evaluation.Eval import *

grammar = Grammar()

grammar.add_rule('START', '', ['EXPR'], 1)
grammar.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1)
grammar.add_rule('EXPR', 'pow_', ['EXPR', 'EXPR'], 1)
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
grammar.add_rule('NUM', 'n', None, 4)


class NumberExpression(LOTHypothesis):

	


    def __init__(self, grammar, domain=100, value=None, f=None, proposal_function=None, **kwargs):
        LOTHypothesis.__init__(self, grammar, proposal_function=proposal_function, **kwargs)
        self.domain = domain

        if value is None:
            self.set_value(grammar.generate('EXPR'), f)
        else:
            self.set_value(value, f)

    def copy(self):
        """ Must define this else we return "FunctionHypothesis" as a copy. We need to return a NumberExpression """
        return NumberExpression(grammar, value=self.value.copy(), prior_temperature=self.prior_temperature)

    def compute_likelihood(self, data):
        """ Computes the likelihood of data. """
        n = len(data)

        for n in range(1,self.domain+1):
        	t = grammar.generate()
        	f = evaluate_expression(t, args=['n'])
        	subset = map(f, range(1,self.domain+1))
        	subset = [item for item in subset if item<=self.domain]

        s = len(subset)
        return (1/s)**n




num_iters = 100
data = [2, 8, 16]


likelihoods = {}

for i in xrange(num_iters):
	t = NumberExpression(grammar)
	likelihoods[t] = t.compute_likelihood(data)

for t in likelihoods.keys():
	# use prior, likelihood to calculate posterior for hypothesis t 
	prior = t.compute_prior()
	likelihood = t.compute_likelihood(data)
	posterior = prior * likelihood

	# normalize
	normalize_param = sum(likelihoods.items() - likelihood)
	posterior = posterior / normalize_param




