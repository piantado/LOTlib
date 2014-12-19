"""
    Simple testing for MCMC methods
"""

from TreeTester import TreeTester
from collections import Counter
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from scipy.stats import chisquare
from LOTlib import lot_iter
from math import exp

NSAMPLES = 10000
SKIP = 10

class GrammarTest(TreeTester):

    def make_h0(self, **kwargs):
        return LOTHypothesis(self.grammar, **kwargs)

    def evaluate_sampler(self, sampler):

        cnt = Counter()
        for h in lot_iter(sampler):
            cnt[h.value] += 1

        ## TODO: When the MCMC methods get cleaned up for how many samples they return, we will assert that we got the right number here
        # assert sum(cnt.values()) == NSAMPLES # Just make sure we aren't using a sampler that returns fewer samples! I'm looking at you, ParallelTempering
        n = sum(cnt.values())
        obsc = [cnt[t] for t in self.trees]
        expc = [exp(t.log_probability())*n for t in self.trees]
        csq, pv = chisquare(obsc, expc)

        self.assertAlmostEqual(sum(obsc), sum(expc))
        assert min(expc) > 5 # or else chisq sux

        for t, c, s in zip(self.trees, obsc, expc):
            print c, s, t
        print (csq, pv), sum(obsc)

        self.assertGreater(pv, 0.0001, msg="Sampler failed chi squared!")

        return csq,pv

    def test_MHSampler(self):
        from LOTlib.Inference.MetropolisHastings import MHSampler
        sampler = MHSampler(self.make_h0(), [], steps=NSAMPLES, skip=SKIP)
        print "MHSampler p value:", self.evaluate_sampler(sampler)


    def test_PriorSampler(self):
        from LOTlib.Inference.PriorSample import PriorSampler
        sampler = PriorSampler(self.make_h0(), [], steps=NSAMPLES)
        print "PriorSampler p value:", self.evaluate_sampler(sampler)

    def test_ParallelTempering(self):
        #TODO: This isn't a strong test of ParallelTempering, since there's no data!
        from LOTlib.Inference.ParallelTempering import ParallelTemperingSampler
        sampler = ParallelTemperingSampler(self.make_h0, [], steps=3*NSAMPLES, skip=SKIP, yield_only_t0=True, temperatures=[1.0, 2.0, 3.0],) ## need more samples in ParallelTempering; NOTE: We could switch how its coded
        print "ParallelTemperingSampler p value:", self.evaluate_sampler(sampler)


    def test_MixtureProposals(self):

        ## Try out MCMC with mixture proposals, to test the forward-back, etc
        from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal
        from LOTlib.Inference.Proposals.InsertDeleteProposal import InsertDeleteProposal
        from LOTlib.Inference.Proposals.InverseInlineProposal import InverseInlineProposal
        from LOTlib.Inference.Proposals.MixtureProposal import MixtureProposal

        p = MixtureProposal([RegenerationProposal(self.grammar),
                            InsertDeleteProposal(self.grammar),
                            InverseInlineProposal(self.grammar)])

        from LOTlib.Inference.MetropolisHastings import MHSampler

        sampler = MHSampler(self.make_h0(proposal_function=p), [], steps=NSAMPLES, skip=SKIP)

        print "MixtureProposal p value:", self.evaluate_sampler(sampler)

    #



if __name__ == '__main__':
    import unittest

    unittest.main()
