from random import randint

from LOTlib.Miscellaneous import Infinity
from LOTlib.Inference.MHShared import MH_acceptance
from LOTlib.Inference.Samplers.MultipleChainMCMC import MultipleChainMCMC


class ParallelTemperingSampler(MultipleChainMCMC):
    """
    Parallel tempering.

    This now includes lots of stats on up-vs-down for tuning the temperatures.
    """

    def __init__(self, make_h0, data, steps=Infinity, temperatures=[1.0, 1.05, 1.15, 1.2], \
                 within_steps=100, swaps=2, yield_only_t0=False, print_swapstats=False, \
                 whichtemperature='likelihood_temperature', **kwargs):

        self.yield_only_t0 = yield_only_t0 #whether we yield all samples, or only from the lowest temperature
        self.within_steps = within_steps
        self.swaps = swaps
        self.print_swapstats = print_swapstats
        self.whichtemperature = whichtemperature

        assert 'nchains' not in kwargs

        MultipleChainMCMC.__init__(self, make_h0, data, nchains=len(temperatures), steps=steps, **kwargs)

        self.temperatures = temperatures

        # and set the temperatures
        for i, t in enumerate(temperatures):
            setattr(self.chains[i], self.whichtemperature, t)

        # Keep track of the number of swaps
        self.upswaps = [0] * (self.nchains-1) # how often are you swapped with the immediately higher chain

        # Keep track of up and down from each chain
        for t in self.chains:
            t.updown = 0 # +1 for up, -1 for down

        self.chains[0].up = 1
        self.chains[len(temperatures)-1].down = -1

        # fraction of particles that are up adn down
        self.nup = [0] * self.nchains
        self.ndown = [0] * self.nchains


    def get_hist(self, smoothed=0.001):
        """
        Return a mildly smoothed histogram
        :return:
        """
        return [ float(a+smoothed)/(float(a+b+2*smoothed)) for a, b in zip(self.nup, self.ndown)]

    def propose_swaps(self):
        """
        Gets called to propose self.swaps number of swaps between chains
        :return:
        """
        for _ in xrange(self.swaps):

            i = randint(0, self.nchains-2)
            cur  = self.chains[i].at_temperature(   self.temperatures[i],   self.whichtemperature) +\
                   self.chains[i+1].at_temperature( self.temperatures[i+1], self.whichtemperature)
            prop = self.chains[i].at_temperature(   self.temperatures[i+1], self.whichtemperature) +\
                   self.chains[i+1].at_temperature( self.temperatures[i],   self.whichtemperature)

            if self.print_swapstats:
                print "# Proposing ", cur-prop, self.upswaps, [ float(a+0.01)/float(a+b+0.01) for a,b in zip(self.nup, self.ndown)]

            if MH_acceptance(cur, prop, 0.0):

                # update the counts
                for idx in [i, i+1]:
                    self.nup[idx]   += (self.chains[idx].updown==1)
                    self.ndown[idx] += (self.chains[idx].updown==-1)

                self.chains[i], self.chains[i+1] = self.chains[i+1], self.chains[i]
                tmp = getattr(self.chains[i], self.whichtemperature)
                setattr(self.chains[i], self.whichtemperature, getattr(self.chains[i+1], self.whichtemperature))
                setattr(self.chains[i+1], self.whichtemperature, tmp)

                self.upswaps[i] += 1

                # keep track of who is up and down
                if i == 0:
                    self.chains[i].updown = 1
                elif i == self.nchains-2:
                    self.chains[self.nchains-1].updown = -1


    def next(self):

        self.nsamples += 1

        self.chain_idx = (self.chain_idx+1) % self.nchains

        if self.nsamples % self.within_steps == 0:
           self.propose_swaps()

        if self.yield_only_t0 and self.chain_idx != 0:
            return self.next() # keep going until we're on the one we yield
            ## TODO: FIX THIS SINCE IT WILL BREAK FOR HUGE NUMBERS OF CHAINS due toi recursion depth
        else:
            return self.chains[self.chain_idx].next()


if __name__ == "__main__":

    from LOTlib import break_ctrlc
    from LOTlib.Miscellaneous import Infinity
    from LOTlib.Examples.ExampleLoader import load_example

    make_hypothesis, make_data = load_example('Number')

    data = make_data(1000)

    z = Z(unique=True)
    tn = TopN(N=10)

    sampler = ParallelTemperingSampler(make_hypothesis, data, steps=100000,
                                       whichtemperature='acceptance_temperature',
                                       temperatures=[1.0, 2., 3., 5., 10., 20.])

    for h in break_ctrlc(tn(z(sampler))):
        # print h.posterior_score, h
        pass

    for x in tn.get_all(sorted=True):
        print x.posterior_score, x

    print z

    print sampler.nup, sampler.ndown
    print sampler.get_hist()