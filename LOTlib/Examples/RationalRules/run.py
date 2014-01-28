"""
	A simple rational rules demo.
	
		A rational analysis of rule-based concept learning. N. D. Goodman, J. B. Tenenbaum, J. Feldman, and T. L. Griffiths (2008). Cognitive Science. 32:1, 108-154. 
		http://www.mit.edu/~ndg/papers/RRfinal3.pdf	
	
	In poor style, this script scatters our imports around to show where each part comes from
"""

import LOTlib

# Many useful functions...
from LOTlib.Miscellaneous import *

# This defines all of our LOT primtiives (is_color_ and is_shape_ below). 
# It needs to be globally defined/imported so that python can eval these functions
from LOTlib.BasicPrimitives import * 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up our grammar. DNF defautly includes the logical connectives
# in disjunctive normal form, but we need to add predicates to it. 

from LOTlib.DefaultGrammars import DNF 
G = DNF

# Two predicates for checking x's color and shape
# Note: per style, functions in the LOT end in _
G.add_rule('PREDICATE', 'is_color_', ['x', 'COLOR'], 1.0)
G.add_rule('PREDICATE', 'is_shape_', ['x', 'SHAPE'], 1.0)

# Some colors/shapes each (for this simple demo)
# These are written in quotes so they can be evaled
G.add_rule('COLOR', q('red'), None, 1.0)
G.add_rule('COLOR', q('blue'), None, 1.0)
G.add_rule('COLOR', q('green'), None, 1.0)
G.add_rule('COLOR', q('mauve'), None, 1.0)

G.add_rule('SHAPE', q('square'), None, 1.0)
G.add_rule('SHAPE', q('circle'), None, 1.0)
G.add_rule('SHAPE', q('triangle'), None, 1.0)
G.add_rule('SHAPE', q('diamond'), None, 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Make up some data
# Let's give data from a simple conjunction (note this example data is not exhaustive)

from LOTlib.DataAndObjects import *

# FunctionData takes a list of arguments and a return value. The arguments are objects (which are handled correctly automatically
# by is_color_ and is_shape_
data = [ FunctionData( [Obj(shape='square', color='red')], True), \
	 FunctionData( [Obj(shape='square', color='blue')], False), \
	 FunctionData( [Obj(shape='triangle', color='blue')], False), \
	 FunctionData( [Obj(shape='triangle', color='red')], False), \
	 ] * 10

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an initial hypothesis
# This is where we set a number of relevant variables -- whether to use RR, alpha, etc. 

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

h0 = LOTHypothesis(grammar=DNF, rrPrior=True, rrAlpha=1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the MH

from LOTlib.Inference.MetropolisHastings import mh_sample

# Run the vanilla sampler. Without steps, it will run infinitely
# this prints out posterior (posterior_score), prior, likelihood, 
for h in mh_sample(h0, data, 10000, skip=100):
	print h.posterior_score, h.prior, h.likelihood, q(h)
	
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This yields data like below. Note that being red is similar to being not-blue, since we have no data on anything other than blue

#-12.6845055552 -10.6327737797 -2.0517317755 "and_( is_color_( x, "red" ), not_( is_shape_( x, "triangle" ) ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( is_color_( x, "red" ), not_( is_shape_( x, "triangle" ) ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( is_color_( x, "red" ), is_shape_( x, "square" ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( is_color_( x, "red" ), not_( is_shape_( x, "triangle" ) ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( is_color_( x, "red" ), is_shape_( x, "square" ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( is_color_( x, "red" ), is_shape_( x, "square" ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( is_color_( x, "red" ), is_shape_( x, "square" ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), is_shape_( x, "square" ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), is_shape_( x, "square" ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( is_color_( x, "red" ), not_( is_shape_( x, "triangle" ) ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( is_color_( x, "red" ), not_( is_shape_( x, "triangle" ) ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( is_color_( x, "red" ), not_( is_shape_( x, "triangle" ) ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), not_( is_shape_( x, "triangle" ) ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), not_( is_shape_( x, "triangle" ) ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), not_( is_shape_( x, "triangle" ) ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( is_color_( x, "red" ), not_( is_shape_( x, "triangle" ) ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( is_color_( x, "red" ), is_shape_( x, "square" ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), not_( is_shape_( x, "triangle" ) ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), not_( is_shape_( x, "triangle" ) ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), not_( is_shape_( x, "triangle" ) ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), not_( is_shape_( x, "triangle" ) ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), is_shape_( x, "square" ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), is_shape_( x, "square" ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), is_shape_( x, "square" ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( not_( is_color_( x, "blue" ) ), not_( is_shape_( x, "triangle" ) ) )"
#-15.6802378288 -13.6285060533 -2.0517317755 "and_( is_color_( x, "red" ), and_( not_( is_shape_( x, "triangle" ) ), is_color_( x, "red" ) ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( is_shape_( x, "square" ), is_color_( x, "red" ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( is_shape_( x, "square" ), is_color_( x, "red" ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( is_shape_( x, "square" ), not_( is_color_( x, "blue" ) ) )"
#-11.9913583747 -9.93962659915 -2.0517317755 "and_( not_( is_shape_( x, "triangle" ) ), not_( is_color_( x, "blue" ) ) )"
#-12.6845055552 -10.6327737797 -2.0517317755 "and_( is_shape_( x, "square" ), not_( is_color_( x, "blue" ) ) )"
