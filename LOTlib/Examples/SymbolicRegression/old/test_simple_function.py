# -*- coding: utf-8 -*-
#import psyco
#psyco.full()

#from closure import *

from bvPCFG import *
from BasicPrimitives import *
from LOTlib.Inference.Samplers.MetropolisHastings import *
from Likelihoods import *
from FiniteSample import *
from Miscellaneous import *
from Lexicon import *
from Objects import *

G = PCFG()
G.add_rule('EXPR', 'plus_', ['EXPR', 'EXPR'], 1.0)
G.add_rule('EXPR', 'times_', ['EXPR', 'EXPR'], 1.0)
#G.add_rule('EXPR', 'exists', ['EXPR', 'EXPR'], 1.0, bvtype='EXPR')
#G.add_rule('EXPR', 'forall', ['EXPR', 'EXPR'], 1.0, bvtype='EXPR')
G.add_rule('EXPR', 'x', [], 5.0) # these terminals should have None for their function type; the literals
G.add_rule('EXPR', '1', [], 1.0)
G.add_rule('EXPR', '2', [], 1.0)
G.add_rule('EXPR', '3', [], 1.0)
G.add_rule('EXPR', '4', [], 1.0)
G.add_rule('EXPR', '5', [], 1.0)

initialh = G.generate('EXPR')
data = [ [1,1], [2,4] ]

##########################################################
#Define a class for running MH
class SimpleExpressionFunction(Hypothesis):

    def __init__(self, v=None):
        Hypothesis.__init__(self)
        if v == None: self.set_value(G.generate('EXPR'))
        else: self.set_value(v)

    def propose(self):

        p = deepcopy(self)
        ph, fb = G.propose(self.value)
        p.set_value(ph)
        return [p, fb]
    def compute_prior(self):
        if self.value.count_subnodes() > 10:
            self.prior = -Infinity
        else: self.prior = self.value.log_probability()
        self.lp = self.prior + self.likelihood
        return self.prior
    def compute_likelihood(self, data):
        self.likelihood = gaussian_likelihood(data,self.value)
        self.lp = self.prior + self.likelihood
        return self.likelihood

    # must wrap these as SimpleExpressionFunctions
    def enumerative_proposer(self):
        for k in G.enumerate_pointwise(self.value):
            yield SimpleExpressionFunction(v=k)

#q = SimpleExpressionFunction()
#d = dict()
#for s in BasicMCMC.tempered_transitions_sample(q, data, 5000, skip=0, temperatures=[1.0, 1.1, 1.2]):
#for s in BasicMCMC.tempered_sample(q, data, 1000):
#for s in BasicMCMC.mh_sample(q, data, 50000, skip=0):

    #print "==>", s

    #for gi in BasicMCMC.gibbs_sample(s, data, steps=1):
        #print "--> ", gi

    #print "\n"

s = SimpleExpressionFunction()
d = dict()
for i in range(1000):

    for si in  LOTlib.MetropolisHastings.mh_sample(s, data, 1, skip=10): s = si

    #print "==>", s

    for gi in  LOTlib.MetropolisHastings.gibbs_sample(s, data, steps=1): s = gi

    hashplus(d, s)#print ph


test_expected_counts(d)
