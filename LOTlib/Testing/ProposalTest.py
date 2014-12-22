"""

TODO: We need to add replicatingrules and apply for the other proposal methods to be tested!

"""


from collections import Counter
from math import exp
from scipy.stats import chisquare

from TreeTesters import InfiniteTreeTester # defines check_tree and setUp

NSAMPLES = 1000

class ProposalTest(InfiniteTreeTester):
    """
    This tests if proposals return well-formed trees.
    """

    def test_RegenerationProposal(self):
        from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal
        rp = RegenerationProposal(self.grammar)

        for tree in self.trees:
            cnt = Counter()
            for _ in xrange(NSAMPLES):
                p, fb = rp.propose_tree(tree)
                cnt[p] += 1

                # Check the proposal
                self.check_tree(p)

            ## check that the proposals are what they should be -- rp.lp_propose is correct!
            obsc = [cnt[t] for t in self.trees]
            expc = [exp(t.log_probability())*n for t in self.trees]
            csq, pv = chisquare([cnt[t] for t in self.trees],
                                [exp(rp.lp_propose(tree, x))*NSAMPLES for x in self.trees])

            # Look at some
            # print ">>>>>>>>>>>", tree
            # for p in self.trees:
            #     print "||||||||||", p
            #     v = rp.lp_propose(tree,p)
            #     print "V=",v

            # for c, e, tt in zip([cnt[t] for t in self.trees],
            #                    [exp(rp.lp_propose(tree, x))*NSAMPLES for x in self.trees],
            #                    self.trees):
            #     print c, e, tt, rp.lp_propose(tree,tt)

            self.assertGreater(pv, 0.001, msg="Sampler failed chi squared!")

    def test_InsertDeleteProposal(self):
        from LOTlib.Inference.Proposals.InsertDeleteProposal import InsertDeleteProposal
        rp = InsertDeleteProposal(self.grammar)

        for tree in self.trees:
            for _ in xrange(100):
                p, fb = rp.propose_tree(tree)
                self.check_tree(p)

if __name__ == '__main__':

    import unittest

    unittest.main()
