"""
    Multiple parallel chains running at once.

    This runs skip steps within each chain, looping to return samples in a roundrobin fashion,
    chain1, chain2, chain3, ...

    Also, steps is the *total* number of steps, not the number of steps for each chain
    This is subclassed by several other inference techniques
"""
from copy import copy

from LOTlib.Miscellaneous import Infinity
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.Inference.Samplers import Sampler

class MultipleChainMCMC(Sampler):
    
    def __init__(self, make_h0, data, steps=Infinity, nchains=10, make_sampler=None, **kwargs):
        """
        :param make_h0: -- a function to make h0 for each chain
        :param data:  -- what data we use
        :param steps:  -- how many steps (total, across all chains)
        :param nchains:  -- how many chains
        :param make_sampler: -- a function that takes make_h0, data, and steps
        :param kwargs: -- special args to sampler
        :return:
        """

        self.nchains = nchains
        self.chain_idx = -1 # what chain are we on? This get incremented before anything, so it starts with 0
        self.nsamples = 0
        assert nchains>0, "Must have > 0 chains specified (you sent %s)"%nchains

        self.chains = [self.make_sampler( make_h0, data, steps=steps/nchains, **kwargs) for _ in xrange(nchains)]

    def make_sampler(self, make_h0, data, **kwargs):
        """
        This is called to make each of our internal samplers. It can be overwritten if you want something fnacy
        :return:
        """
        return MHSampler( make_h0(), data, **kwargs)

    def __iter__(self):
        return self

    def next(self):
        self.nsamples += 1
        self.chain_idx = (self.chain_idx+1) % self.nchains
        return self.chains[self.chain_idx].next()

    def reset_counters(self):
        for c in self.chains:
            c.reset_counters()

    def acceptance_ratio(self):
        """
            Return the mean acceptance rate of all chains
        """
        return [c.acceptance_ratio() for c in self.chains]

    def set_state(self, s, **kwargs):
        """
        Set the state of this sampler. By necessity, we set the states of all samplers to copies. This is required when, for instance, we
        next parallel tempering within PartitionMCMC
        :param s: the state we set
        """
        for c in self.chains:
            c.set_state(copy(s), **kwargs) # it had better be a copy or all hell breaks loose

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    from LOTlib import break_ctrlc
    from LOTlib.Examples.ExampleLoader import load_example

    make_hypothesis, make_data = load_example('Number')
    data = make_data(300)

    for h in break_ctrlc(MultipleChainMCMC(make_hypothesis, data, steps=2000, nchains=10)):
        print h.posterior_score, h
