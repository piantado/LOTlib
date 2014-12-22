"""
    Simple testing for MCMC methods
"""

from TreeTesters import FiniteTreeTester, InfiniteTreeTester


NSAMPLES = 10000
SKIP = 5

class GrammarTest(FiniteTreeTester):

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


class MCMCProposalTest(InfiniteTreeTester):
    """
    This test MCMC under different proposals
    """

    def test_RegenerationProposals(self):
        from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal
        p = RegenerationProposal(self.grammar)

        from LOTlib.Inference.MetropolisHastings import MHSampler

        sampler = MHSampler(self.make_h0(proposal_function=p), [], steps=NSAMPLES, skip=SKIP)

        # Here we plot them, since chisq won't work well with many zeros
        # TODO: Implement a better statistical test
        print "Regeneration proposal"
        self.plot_sampler('regeneration.png',sampler)
        print "-----------------------------------------------------"


    def test_InsertDeleteProposals(self):
        from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal
        from LOTlib.Inference.Proposals.InsertDeleteProposal import InsertDeleteProposal
        from LOTlib.Inference.Proposals.MixtureProposal import MixtureProposal

        p = MixtureProposal([RegenerationProposal(self.grammar),
                             InsertDeleteProposal(self.grammar)])

        from LOTlib.Inference.MetropolisHastings import MHSampler

        # Here we plot them, since chisq won't work well with many zeros
        # TODO: Implement a better statistical test
        sampler = MHSampler(self.make_h0(proposal_function=p), [], steps=NSAMPLES, skip=SKIP)

        print "InsertDeleteProposal"
        self.plot_sampler('insert-delete.png', sampler)
        print "-----------------------------------------------------"

    def test_InverseInlineProposal(self):
        from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal
        from LOTlib.Inference.Proposals.InverseInlineProposal import InverseInlineProposal
        from LOTlib.Inference.Proposals.MixtureProposal import MixtureProposal

        p = MixtureProposal([RegenerationProposal(self.grammar),
                             InverseInlineProposal(self.grammar)])

        from LOTlib.Inference.MetropolisHastings import MHSampler

        # Here we plot them, since chisq won't work well with many zeros
        # TODO: Implement a better statistical test
        sampler = MHSampler(self.make_h0(proposal_function=p), [], steps=NSAMPLES, skip=SKIP)

        print "Inverse Inline proposal"
        self.plot_sampler('inverse-inline.png', sampler)
        print "-----------------------------------------------------"

if __name__ == '__main__':
    import unittest

    unittest.main()