
"""
	An example using first order logic (to test out functions on bound variables)
"""

from LOTlib.Grammar import *
from LOTlib.FiniteBestSet import *
from LOTlib.Hypothesis import *

G = Grammar()
G.BV_WEIGHT = 2.0 # probability weight for our introduced variables

G.add_rule('BOOL', 'x', None, 2.0)

G.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
G.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
G.add_rule('BOOL', 'not_', ['BOOL'], 1.0)

G.add_rule('BOOL', 'exists_', ['FUNCTION', 'SET'], 0.50)
G.add_rule('BOOL', 'forall_', ['FUNCTION', 'SET'], 0.50) 

G.add_rule('SET', 'S', [], 1.0)

G.add_rule('FUNCTION', 'lambda', ['BOOL'], 1.0, bv_name='BOOL', bv_args=None) # bvtype means we introduce a bound variable below

## Generate some and print out unique ones
upq = FiniteBestSet()
for i in xrange(1000):
	x = G.generate('BOOL')
	
	if x not in upq:
		upq.push(x,0.0)
		print x
		print x.log_probability(), x
		
#for i in xrange(10000):
	#x = G.generate('BOOL')
	
	#print x
	
	#for k in xrange(10):
		#y = G.propose(x)
		#print "\t", y[0].log_probability(), y

#for k in G.increment_tree("BOOL", 5):
	#print k