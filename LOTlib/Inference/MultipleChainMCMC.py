"""
    Multiple parallel chains running at once.

    This runs skip steps within each chain, looping to return samples in a roundrobin fashion,
    chain1, chain2, chain3, ...

    Also, steps is the *total* number of steps, not the number of steps for each chain
    This is subclassed by several other inference techniques
"""
from LOTlib.Miscellaneous import Infinity
from MetropolisHastings import MHSampler
from Sampler import Sampler


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
        ## TODO: HANDLE THE CASE WHERE STEPS IS NOT DIVISIBLE BY CHAINS

        if make_sampler is None:
            make_sampler = lambda mh0, data, **kwargs: MHSampler( make_h0(), data, steps=steps/nchains, **kwargs)

        self.chains = [make_sampler( make_h0, data, **kwargs) for _ in xrange(nchains)]

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    from LOTlib.Examples.Number import generate_data, NumberExpression, grammar
    data = generate_data(300)

    make_h0 = lambda : NumberExpression(grammar)

    sampler = MultipleChainMCMC(make_h0, data, steps=2000, nchains=100)
    for h in sampler:
        print h.posterior_score, h
