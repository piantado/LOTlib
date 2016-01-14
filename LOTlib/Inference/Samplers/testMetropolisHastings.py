

import unittest
from collections import Counter
from numpy import log, exp
from scipy.stats import chisquare

from LOTlib import break_ctrlc
from LOTlib.Miscellaneous import logsumexp, attrmem
from MetropolisHastings import MHSampler



class TestMetropolisHastings(unittest.TestCase):
    """
    Test the sampler, using a default grammar

    TODO: Add tree checks to samples from the sampler!

    """
    def runTest(self):
        NSAMPLES = 10000

        from LOTlib.DefaultGrammars import finiteTestGrammar as grammar

        from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
        class MyH(LOTHypothesis):

            @attrmem('likelihood')
            def compute_likelihood(self, *args, **kwargs):
                return 0.0

            @attrmem('prior')
            def compute_prior(self):
                return grammar.log_probability(self.value)

        print "# Taking MHSampler for a test run"
        cnt = Counter()
        h0 = MyH(grammar=grammar)
        for h in break_ctrlc(MHSampler(h0, [], steps=NSAMPLES, skip=10)): # huh the skip here seems to be important
            cnt[h] += 1
        trees = list(cnt.keys())
        print "# Done taking MHSampler for a test run"

        ## TODO: When the MCMC methods get cleaned up for how many samples they return, we will assert that we got the right number here
        # assert sum(cnt.values()) == NSAMPLES # Just make sure we aren't using a sampler that returns fewer samples! I'm looking at you, ParallelTempering

        Z = logsumexp([grammar.log_probability(t.value) for t in trees]) # renormalize to the trees in self.trees
        obsc = [cnt[t] for t in trees]
        expc = [exp( grammar.log_probability(t.value))*sum(obsc) for t in trees]

        # And plot here
        expc, obsc, trees = zip(*sorted(zip(expc, obsc, trees), reverse=True))
        import matplotlib.pyplot as plt
        plt.subplot(111)
        # Log here spaces things out at the high end, where we can see it!
        plt.scatter(log(range(len(trees))), expc, color="red", alpha=1.)
        plt.scatter(log(range(len(trees))), obsc, color="blue", marker="x", alpha=1.)
        plt.savefig('finite-sampler-test.pdf')
        plt.clf()

        # Do chi squared test
        csq, pv = chisquare(obsc, expc)
        self.assertAlmostEqual(sum(obsc), sum(expc))

        # And examine
        for t, c, s in zip(trees, obsc, expc):
            print c, s, t
        print (csq, pv), sum(obsc)

        self.assertGreater(pv, 0.01, msg="Sampler failed chi squared!")





# class TestMetropolisHastings2(unittest.TestCase):
#     Test the sampler, using the number model
    # def runTest(self):
    #
    #     from LOTlib.Examples.Number.Model import make_data, make_hypothesis
    #
    #     NSAMPLES = 1000
    #
    #     cnt = Counter()
    #     for h in MHSampler(make_hypothesis(),  make_data(100), steps=NSAMPLES):
    #         cnt[h] += 1
    #     trees = list(cnt.keys())
    #
    #     self.assertEqual(sum(cnt.values()), NSAMPLES, 'sampler did not return the correct number of samples')
    #
    #     Z = logsumexp([t.posterior_score for t in trees]) # renormalize to the trees in self.trees
    #     obsc = [cnt[t] for t in trees]
    #     expc = [exp(t.posterior_score-Z)*sum(obsc) for t in trees]
    #     csq, pv = chisquare(obsc, expc)
    #     self.assertAlmostEqual(sum(obsc), sum(expc))
    #
    #     for c, s, t in zip(obsc, expc, trees):
    #         print c, s, t, t.posterior_score, Z
    #     print (csq, pv), sum(obsc)
    #
    #     self.assertGreater(pv, 0.05, msg="Sampler failed chi squared!")
    #
    #     return csq, pv


