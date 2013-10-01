# -*- coding: utf-8 -*-

from LOTlib.PCFG import PCFG
from LOTlib.BasicPrimitives import *
import LOTlib.MetropolisHastings
from LOTlib.FiniteBestSet import FiniteBestSet
from LOTlib.Miscellaneous import *
from LOTlib.Lexicon import *
from LOTlib.DataAndObjects import *
from LOTlib.Hypothesis import *

CHAINS = 10
STEPS = 100000
SKIP = 0
PRIOR_TEMPERATURE=1.0

data = {
	('NP','VP') : 0.5,
	('VP','NP') : 0.5
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

G = PCFG()
G.add_rule('START', '', ['EXPR'], 1.0)

G.add_rule('BOOL', 'and_',    ['BOOL', 'BOOL'], 1.0)
G.add_rule('BOOL', 'or_',     ['BOOL', 'BOOL'], 1.0)
G.add_rule('BOOL', 'not_',    ['BOOL'], 1.0)

#G.add_rule('EXPR', 'if_',    ['BOOL', 'EXPR', 'EXPR'], 1.0)
#G.add_rule('BOOL', 'equal_',    ['EXPR', 'EXPR'], 1.0)

#G.add_rule('BOOL', 'flip_', [None], 4.0)
#G.add_rule('NOTHING', 'dummy_', [], 1.0) # don't know how to make it call a fn with no args

#G.add_rule('BOOL', 'True',    [], 1.0)
#G.add_rule('BOOL', 'False',   [], 1.0)

G.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 1.)
G.add_rule('EXPR', 'cdr_',  ['EXPR'], 1.)
G.add_rule('EXPR', 'car_',  ['EXPR'], 1.)

G.add_rule('EXPR', 'NP_', [], 2.0)
G.add_rule('EXPR', 'VP_', [], 2.0)


# HEre's a way to do it without EXPR(), but it's ugly:
#G.add_rule('EXPR', 'apply_',  ['FUNCTION', 'THUNK'], 1.)
#G.add_rule('FUNCTION', 'lambda',  ['EXPR'], 1., bv=['PRIMITIVETHUNK'])
#G.add_rule('EXPR', 'apply_',  ['PRIMITIVETHUNK'], 1.,)
#G.add_rule('THUNK', 'lambda',  ['EXPR'], 1., bv=[])
#G.add_rule('PRIMITIVETHUNK', 'lambdaNull',  [], 0.00001) ## And just so this is defined



G.add_rule('EXPR', 'apply_',  ['FUNCTION', 'THUNK'], 1.)
G.add_rule('FUNCTION', 'lambda',  ['EXPR'], 1., bv=['EXPR()'])
G.add_rule('THUNK', 'lambda',  ['EXPR'], 1., bv=[])



# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# We store the top 100 from each run
finitesample = FiniteBestSet(100, max=True) 
	
	
for i in xrange(10000):
	x = StructuralHypothesis(G)
	
	print "Hypothesis:\t", x
	print "Evaled:\t", x(None)
	print "\n\n"
	
	