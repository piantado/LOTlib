
"""
	A LOTlib example for bound variables.
	
	This does inference over cons, cdr, car expressions. 
	NOTE: it does not work very well for complex functions since it is hard to sample a close function -- not much of a gradient to climb on cons, cdr, car
	
"""

import LOTlib.MetropolisHastings
from Shared import *

STEPS = 1000000
ALPHA = 0.9

# Make up some data 

# here just doubling x :-> cons(x,x)
data = [
	FunctionData( args=[ [] ], output=[[],[]] ),
	FunctionData( args=[ [[]] ], output=[[[]], [[]]] ),
       ]
       
 # A little more interesting. Squaring: N parens go to N^2
#data = [
	#FunctionData( args=[[  [ [] ] * i  ]], output=[ [] ] * (i**2) ) \
	#for i in xrange(1,10)
       #]  
       
# And run
h0 = SchemeFunction(G, ALPHA=ALPHA)       
for x in LOTlib.MetropolisHastings.mh_sample(h0, data, STEPS):
	
	print x.lp, x
	for di in data:
		print "\t", di.args, "->", x(*di.args), " ; should be ", di.output
	