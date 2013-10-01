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

G.add_rule('EXPR', 'if_',    ['BOOL', 'EXPR', 'EXPR'], 1.0)
G.add_rule('BOOL', 'equal_',    ['EXPR', 'EXPR'], 1.0)

G.add_rule('BOOL', 'flip_', ['NOTHING'], 4.0)
G.add_rule('NOTHING', 'dummy_', [], 1.0) # don't know how to make it call a fn with no args

G.add_rule('BOOL', 'True',    [], 1.0)
G.add_rule('BOOL', 'False',   [], 1.0)

G.add_rule('EXPR', 'cons_', ['EXPR', 'EXPR'], 2.0)
G.add_rule('EXPR', 'cdr_',  ['EXPR'], 0.5)
G.add_rule('EXPR', 'car_',  ['EXPR'], 0.5)

G.add_rule('EXPR', 'NP_', [], 4.0)
G.add_rule('EXPR', 'VP_', [], 4.0)

## note that this can take basically any types for return values
#G.add_rule('WORD', 'if_',    ['BOOL', 'WORD', 'WORD'], 0.5)
#G.add_rule('WORD', 'ifU_',    ['BOOL', 'WORD'], 0.5) # if returning undef if condition not met
#
#G.add_rule('BOOL', 'equal_',    ['WORD', 'WORD'], 1.0)
#G.add_rule('WORD', 'L_',        ['SET'], 1.0) 

#G.add_rule('SET', 'x',     [], 10.0)

#G.add_rule('SET', 'union_',     ['SET', 'SET'], 1.0)
#G.add_rule('SET', 'intersection_',     ['SET', 'SET'], 1.0)
#G.add_rule('SET', 'setdifference_',     ['SET', 'SET'], 1.0)
#G.add_rule('SET', 'select_',     ['SET'], 1.0)
#
#G.add_rule('WORD', 'next_', ['WORD'], 1.0)
#G.add_rule('WORD', 'prev_', ['WORD'], 1.0)

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# We store the top 100 from each run
finitesample = FiniteBestSet(100, max=True) 
	
	
initial_hyp = StructuralHypothesis(G)
best = initial_hyp


# populate the finite sample by running the sampler for this many steps
for x in LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, STEPS, skip=SKIP):
	finitesample.push(x, x.lp)
	if x.lp > best.lp:
		best = x
		print x.lp, x.prior, x.likelihood, q(x)

## and display
for r in finitesample.get_all(decreasing=False, sorted=True):
	print round(r.lp,1), "\t", round(r.prior,1), "\t", round(r.likelihood,1), "\t", q(str(r)), "\t", r.counts 
	
	
