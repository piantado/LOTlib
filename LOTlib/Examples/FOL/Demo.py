
"""
An example of inference over first-order logical expressions.
Here, we take sets of objects and generate quantified descriptions
"""
from LOTlib import lot_iter
from Model import *
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Just generate from this grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#for i in xrange(100):
        #print grammar.generate()

# Or we can make them as hypotheses (functions of S):
#for i in xrange(100):
        #print LOTHypothesis(grammar, args=['S'])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Or real inference:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.DataAndObjects import FunctionData, Obj # for nicely managing data
from LOTlib.Inference.MetropolisHastings import MHSampler # for running MCMC

# Make up some data -- here just one set containing {red, red, green} colors
data = [ FunctionData(input=[ {Obj(color='red'), Obj(color='red'), Obj(color='green')} ], output=True) ]

# Create an initial hypothesis -- defaultly generated at random from the grammar
h0 = LOTHypothesis(grammar, args=['S'])

if __name__ == "__main__":
        
    for h in lot_iter(MHSampler(h0, data, steps=4000)): # run sampler
        print h.likelihood, h.prior, h.posterior_score, h
