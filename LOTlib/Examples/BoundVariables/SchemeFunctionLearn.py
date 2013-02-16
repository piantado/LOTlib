
"""
	A LOTlib example for bound variables.
	
	This does inference over cons, cdr, car expressions. 
	NOTE: it does not work very well for complex functions since it is hard to sample a close function -- not much of a gradient to climb on cons, cdr, car
	
"""

import LOTlib.MetropolisHastings
from LOTlib.PCFG import PCFG
from LOTlib.Hypothesis import *

ALPHA = 0.90
LL_TEMPERATURE = 0.1
STEPS = 1000000
SKIP = 0

G = PCFG()

# A very simple version of lambda calculus
G.add_rule('EXPR', 'apply_', ['FUNC', 'EXPR'], 1.0)
G.add_rule('EXPR', 'x', [], 5.0)
G.add_rule('FUNC', 'lambda', ['EXPR'], 1.0, bv=['EXPR'])

G.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.0)
G.add_rule('EXPR', 'cdr_',  ['EXPR'], 1.0)
G.add_rule('EXPR', 'car_',  ['EXPR'], 1.0)

G.add_rule('EXPR', '[]',  [], 1.0)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## Wrap the FunctionNode in a hypothesis, that computes prior and likelihood, etc. 
class SchemeHypothesis(Hypothesis):
	
	def __init__(self, v=None): 
		Hypothesis.__init__(self)
		if v == None: self.set_value(G.generate('EXPR'))
		else: self.set_value(v)
		
	def propose(self): 
		p = deepcopy(self)
		ph, fb = G.propose(self.value)
		p.set_value(ph)
		return [p, fb]
	def compute_prior(self): 
		if self.value.count_subnodes() > 25:
			self.prior = -Infinity
		else: self.prior = self.value.log_probability()
		self.lp = self.prior + self.likelihood
		return self.prior
	def compute_likelihood(self, data):
		
		f = evaluate_expression(self.value)
		
		ll = 0.0 # the log likelihood
		
		for di in data:
			frm,to = di
			#print frm, to
			if str(f(frm)) == str(to):  ll += log(ALPHA)
			else:                       ll += log(1.0-ALPHA)
			
		self.likelihood = ll / LL_TEMPERATURE
		self.lp = self.prior + self.likelihood
		return self.likelihood
		
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The sampling run

data = [
	( [], [[],[]] ), ## (from, to) 
	( [[]], [[[]], [[]]] )
       ]
       
initial_hyp = SchemeHypothesis()
       
for x in LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, STEPS, skip=SKIP):
	
	print x.lp, x
	f = evaluate_expression(x)
	for frm,to in data:
		print "\t", frm, "->", f(frm), " ; should be ", to
	