from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Inference.MetropolisHastings import mh_sample
from math import log

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A simple grammar for scheme, including lambda

G = Grammar()

# A very simple version of lambda calculus
G.add_rule('START', '', ['EXPR'], 1.0)
G.add_rule('EXPR', 'apply_', ['FUNC', 'EXPR'], 1.0)
G.add_rule('EXPR', 'x', None, 5.0)
G.add_rule('FUNC', 'lambda', ['EXPR'], 1.0, bv_name='EXPR', bv_args=None)

G.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.0)
G.add_rule('EXPR', 'cdr_',  ['EXPR'], 1.0)
G.add_rule('EXPR', 'car_',  ['EXPR'], 1.0)

G.add_rule('EXPR', '[]',  None, 1.0)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A class for scheme hypotheses that just computes the input/output pairs with the appropriate probability
class SchemeFunction(LOTHypothesis):
	
	# Prior, proposals, __init__ are all inherited from LOTHypothesis
	
	def compute_likelihood(self, data):
		
		self.likelihood = 0.0 # the log likelihood
		
		for di in data:
			# We'll just use a string comparison on outputs here
			if str(self(*di.args)) == str(di.output):  
				self.likelihood += log(self.ALPHA)
			else:                       
				self.likelihood += log(1.0-self.ALPHA)
			
		self.lp = self.prior + self.likelihood
		return self.likelihood