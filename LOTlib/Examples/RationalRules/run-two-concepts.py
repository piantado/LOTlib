"""
	Rational rules over two concepts at the same time.
	
	Another way to do this would be to use a Lexicon and write a custom likelihood method
"""

from Shared import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# somewhat weirdly, we'll make an upper node above "START" for the two concepts
# and require it to check if concept (an argument below) is 'A'
G.add_rule('TWO_CONCEPT_START', 'if_', ['(concept==\'A\')', 'START', 'START'], 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We need data in a different format for this guy

from LOTlib.DataAndObjects import *

# The argumetns are [concept,object]
data = [ FunctionData( ['A', Obj(shape='square', color='red')],    True), 
	 FunctionData( ['A', Obj(shape='square', color='blue')],   False), 
	 FunctionData( ['A', Obj(shape='triangle', color='blue')], False), 
	 FunctionData( ['A', Obj(shape='triangle', color='red')],  False), 
	 
	 FunctionData( ['B', Obj(shape='square', color='red')],    False), 
	 FunctionData( ['B', Obj(shape='square', color='blue')],   True), 
	 FunctionData( ['B', Obj(shape='triangle', color='blue')], True), 
	 FunctionData( ['B', Obj(shape='triangle', color='red')],  True)] * 10 # How many data points exactly like these?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an initial hypothesis
# This is where we set a number of relevant variables -- whether to use RR, alpha, etc. 


from LOTlib.Hypotheses.RationalRulesLOTHypothesis import RationalRulesLOTHypothesis

# Here we give args as "concept" (used in TWO_CONCEPT_START above) and "x"
h0 = RationalRulesLOTHypothesis(grammar=G, rrAlpha=1.0, ALPHA=0.9, start='TWO_CONCEPT_START', args=['concept', 'x'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the MH

from LOTlib.Inference.MetropolisHastings import mh_sample

# Run the vanilla sampler. Without steps, it will run infinitely
# this prints out posterior (posterior_score), prior, likelihood, 
for h in mh_sample(h0, data, 10000, skip=100):
	print h.posterior_score, h.prior, h.likelihood, q(h)
