"""
	A very simple case of predicate invention, inspired by 
	
	T. D. Ullman, N. D. Goodman and J. B. Tenenbaum (2012), Theory learning as stochastic search in the language of thought. Cognitive Development. 
	
	Here, we invent simple predicates whose value is determined by a set membership (BASE-SET), and express logical
	concepts over those predicates. Data is set up to be like magnetism, with positives (pi) and negatives (ni) that interact
	with each other but not within groups
	
	TODO: Let's add another class--the non-magnetic ones!
	
"""

import LOTlib
from LOTlib.Miscellaneous import unique, qq
from LOTlib.Grammar import Grammar
from LOTlib.DataAndObjects import FunctionData
from LOTlib.BasicPrimitives import *
from LOTlib.FunctionNode import cleanFunctionNodeString
import itertools

G = Grammar()

G.add_rule('START', '', ['Pabstraction'], 1.0) # a predicate abstraction

# lambdaUsePredicate is where you can use the predicate defined in lambdaDefinePredicate
G.add_rule('Pabstraction',  'apply_', ['lambdaUsePredicate', 'lambdaDefinePredicate'], 1.0, )
G.add_rule('lambdaUsePredicate', 'lambda', ['INNER-BOOL'],    5.0, bv_type='INNER-BOOL', bv_args=['OBJECT'], bv_prefix='F')
G.add_rule('lambdaUsePredicate', 'lambda', ['Pabstraction'], 1.0,  bv_type='INNER-BOOL', bv_args=['OBJECT'], bv_prefix='F')

# Define a predicate that will just check if something is in a BASE-SET
G.add_rule('lambdaDefinePredicate', 'lambda', ['lambdaDefinePredicateINNER'], 1.0,  bv_type='OBJECT', bv_args=None, bv_prefix='z')
# the function on objects, that allows them to be put into classes (analogous to a logical model here)
G.add_rule('lambdaDefinePredicateINNER', 'is_in_', ['OBJECT', 'BASE-SET'], 1.0)

# After we've defined F, these are used to construct the concept
G.add_rule('INNER-BOOL', 'and_', ['INNER-BOOL', 'INNER-BOOL'], 1.0)
G.add_rule('INNER-BOOL', 'or_', ['INNER-BOOL', 'INNER-BOOL'], 1.0)
G.add_rule('INNER-BOOL', 'not_', ['INNER-BOOL'], 1.0)

G.add_rule('OBJECT', 'x', None, 1.0)
G.add_rule('OBJECT', 'y', None, 1.0)

# BASE-SET is here a set of BASE-OBJECTS (non-args)
G.add_rule('BASE-SET', 'set_add_', ['BASE-OBJECT', 'BASE-SET'], 1.0)
G.add_rule('BASE-SET', 'set_', [], 1.0)

objects = [ t+str(i) for t,i in itertools.product('pnx', range(3)) ] 

for o in objects:
	G.add_rule('BASE-OBJECT', qq(o), None, 1.0)

#from LOTlib.Subtrees import *
#for t in generate_trees(G):
	#print t
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up data -- true output means attraction (p=positive; n=negative)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data = []

for a,b in itertools.product(objects, objects):
	
	myinput  = [a,b]
	
	# opposites (n/p) interact; x interacts with nothing
	myoutput = (a[0] != b[0]) and (a[0] != 'x') and (b[0] != 'x')
	
	data.append( FunctionData(input=myinput, output=myoutput) )
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run mcmc
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Proposals import *
mp = MixtureProposal(G, [RegenerationProposal(G), InsertDeleteProposal(G)] )

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
h0 = LOTHypothesis(G, args=['x', 'y'], ALPHA=0.999, proposal_function=mp) # alpha here trades off with the amount of data. Currently assuming no noise, but that's not necessary

from LOTlib.Inference.MetropolisHastings import mh_sample
for h in mh_sample(h0, data, 4000000, skip=100):
	print h.posterior_score, h.likelihood, h.prior, cleanFunctionNodeString(h)
	#print map( lambda d: h(*d.input), data)
	#print "\n"

	