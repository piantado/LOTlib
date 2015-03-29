from copy import copy

from numpy import median
from numpy.random import normal, randint
from MultipleChainMCMC import MultipleChainMCMC
from LOTlib import break_ctrlc
from LOTlib.Miscellaneous import logsumexp, q, Infinity, logplusexp
from copy import copy


class ParticleSwarm(MultipleChainMCMC):
    """
    Make a swarm of particles with different temperatures, inference parameters. The default number comes from MultipleChainMCMC
    Keep the good ones, as determined by those which find the most new space

    TODO: - It might be better to add my_add incrementally, so that the earlier chains
          will prevent others from duplicating?
          - Maybe make resampling relative to the amount of discovered probability mass?
    """

    def __init__(self, make_h0, data, within_steps=100, **kwargs):

        MultipleChainMCMC.__init__(self, make_h0, data, **kwargs)

        self.within_steps = within_steps

        self.kwargs = kwargs
        self.seen = set()
        self.chainZ = [-Infinity for _ in xrange(self.nchains)]
        self.nsamples = 0  # How many samples have we drawn?

    def refresh(self):
        """
        This gets called when we have run within_steps to refresh some of the chains. It get overwritten in the inherited classes

        In this version, we take the bottom half and copy random ones of the top half
        """

        ranks = sorted(range(self.nchains), key=lambda x: self.chainZ[x])
        self.chains = [self.chains[r] for r in ranks]  # re-order by scores

        for i in range(self.nchains / 2):  # take the worst
            j = randint(0, self.nchains / 2) + self.nchains / 2 - 1  # copy a random better one
            self.chains[i].current_sample = copy(self.chains[j].current_sample)

        # And reset this
        self.chainZ = [-Infinity for _ in xrange(self.nchains)]


    def next(self):

        nxt = MultipleChainMCMC.next(self)  # get the next one
        idx = self.chain_idx
        if nxt not in self.seen:
            self.chainZ[idx] = logplusexp(self.chainZ[idx], nxt.posterior_score)
            self.seen.add(nxt)

            # # Process the situation where we need to re-organize
        if self.nsamples % (self.within_steps * self.nchains) == 0 and self.nsamples > 0:
            self.refresh()

        self.nsamples += 1

        return nxt


class ParticleSwarmPriorResample(ParticleSwarm):
    """
    Like ParticleSwarm, but resamples from the prior
    """

    def refresh(self):
        """
            Resample by resampling those below the median from the prior.
        """
        m = median(self.chainZ)

        for i in range(self.nchains):
            if self.chainZ[i] < m:
                self.chains[i] = self.make_h0(**self.kwargs)
            self.chainZ[i] = -Infinity  # reset this


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~			
if __name__ == "__main__":
    from LOTlib.Examples.Number.Global import generate_data, make_h0

    data = generate_data(300)

    ps = ParticleSwarm(make_h0, data)
    for h in break_ctrlc(ps):
        print h.posterior_score, h

        if len(ps.seen) > 0:
            print "#", sorted(ps.seen, key=lambda x: x.posterior_score, reverse=True)[0]
