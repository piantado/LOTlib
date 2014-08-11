"""
        A version that cares more about recent data, showing how to use Hypotheses.DecayedLikelihoodHypothesis
"""

from Shared import *

G = grammar
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an initial hypothesis
# This is where we set a number of relevant variables -- whether to use RR, alpha, etc.

from LOTlib.Hypotheses.DecayedLikelihoodHypothesis import DecayedLikelihoodHypothesis
from LOTlib.Hypotheses.RationalRulesLOTHypothesis import RationalRulesLOTHypothesis

# Define our class solely throuhg inheritance of these two.
# Here, DecayedLikelihoodHypothesis provides the decaying functions, RationalRulesLOTHypothesis provides everything else
# We just need to pass the initializer to both superclasses
class MyHypothesis(DecayedLikelihoodHypothesis, RationalRulesLOTHypothesis):
    def __init__(self, grammar, value=None, ll_decay=0.0, args=('x',), **kwargs ):
        DecayedLikelihoodHypothesis.__init__(self, value=None, ll_decay=ll_decay)
        RationalRulesLOTHypothesis.__init__(self, grammar, value=value, args=args, **kwargs) # args will just get passed to RationalRulesLOTHypothesis

h0 = MyHypothesis(G, ll_decay=1.0, rrAlpha=1.0, args=['x'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the MH

from LOTlib.Inference.MetropolisHastings import mh_sample

# Run the vanilla sampler. Without steps, it will run infinitely
# this prints out posterior (posterior_score), prior, likelihood,
for h in mh_sample(h0, data, 10000, skip=100):
    print h.posterior_score, h.prior, h.likelihood, q(h)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This setup requires the *later* data to be upweighted, meaning that hypotheses that get
# later data wrong should be given lower likelhood. But also with the decay, the overall
# magnitude of the likelihood decreases. As in:

#-12.2035235691 -9.93962659915 -2.26389696996 'and_(is_shape_(x, 'triangle'), is_color_(x, 'blue'))'
#-10.7313040795 -9.93962659915 -0.791677480378 'and_(is_shape_(x, 'triangle'), is_color_(x, 'red'))'
