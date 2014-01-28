
"""
	A LOTlib example for bound variables.
	
	This does inference over cons, cdr, car expressions. 
	NOTE: it does not work very well for complex functions since it is hard to sample a close function -- not much of a gradient to climb on cons, cdr, car
	
"""

from Shared import *

STEPS = 1000000
ALPHA = 0.9

# Make up some data 

# here just doubling x :-> cons(x,x)
data = [
	FunctionData( input=[ [] ],   output=[[],[]] ),
	FunctionData( input=[ [[]] ], output=[[[]], [[]]] ),
       ]
       
 # A little more interesting. Squaring: N parens go to N^2
#data = [
	#FunctionData( input=[[  [ [] ] * i  ]], output=[ [] ] * (i**2) ) \
	#for i in xrange(1,10)
       #]  
       
# And run
h0 = SchemeFunction(G, ALPHA=ALPHA)       
for x in mh_sample(h0, data, STEPS):
	
	print x.lp, x
	for di in data:
		print "\t", di.input, "->", x(*di.input), " ; should be ", di.output
	