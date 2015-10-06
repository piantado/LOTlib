from scipy import interpolate

from ParallelTempering import ParallelTemperingSampler


class AdaptiveParallelTemperingSampler(ParallelTemperingSampler):
    """
    Adaptive setting of the temperatures via

    Katzgraber, H. G., Trebst, S., Huse, D. A., & Troyer, M. (2006). Feedback-optimized parallel tempering monte carlo. Journal of Statistical Mechanics: Theory and Experiment, 2006, P03018
    """

    def __init__(self, make_h0, data, adapt_at=[50000, 100000, 200000, 300000, 500000, 1000000], **kwargs):

        ParallelTemperingSampler.__init__(self, make_h0, data, **kwargs)
        self.adapt_at = adapt_at


    def adapt_temperatures(self, epsilon=0.001):
        """
        Adapat our temperatures, given self.nup and self.ndown
        This follows ComputeAdaptedTemperatures in https://github.com/stuhlmueller/mcnets/blob/master/mcnets/tempering.py
        :return:
        """
        hist = self.get_hist()

        linear_hist = [x/float(self.nchains-1) for x in reversed(range(self.nchains))]

        monotonic_hist = [x*float(1.-epsilon) + y*epsilon for x, y in zip(hist, linear_hist)]

        # print "Linear:", linear_hist
        # print "Monotonic:", monotonic_hist

        # Hmm force monotonic to have 0,1?
        monotonic_hist[0], monotonic_hist[-1] = 1.0, 0.0

        f = interpolate.interp1d(list(reversed(monotonic_hist)), list(reversed(self.temperatures)))

        newt = [self.temperatures[0]]
        for i in reversed(range(2, self.nchains)):
            # print i, float(i-1) / (self.nchains-1), frac(float(i-1) / (self.nchains-1))
            newt.append(f([float(i-1.) / (self.nchains-1)])[0])

        # keep the old temps
        newt.append(self.temperatures[-1])

        self.temperatures = newt

        print "# Adapting temperatures to ", self.temperatures
        print "# Acceptance ratio:", self.acceptance_ratio()

        # And set each temperature chain
        for c, t in zip(self.chains, self.temperatures):
            c.likelihod_temperature = t


    def next(self):
        ret = ParallelTemperingSampler.next(self)

        if self.nsamples in self.adapt_at: ## TODO: Maybe make this faster?
            self.adapt_temperatures()

        return ret


if __name__ == "__main__":

    from LOTlib import break_ctrlc
    from LOTlib.Examples.Number2015.Model import generate_data, make_h0
    data = generate_data(300)

    from LOTlib.TopN import TopN
    z = Z(unique=True)
    tn = TopN(N=10)

    from LOTlib.Miscellaneous import logrange

    sampler = AdaptiveParallelTemperingSampler(make_h0, data, steps=1000000, \
                                               yield_only_t0=False, whichtemperature='acceptance_temperature', \
                                               temperatures=logrange(1.0, 10.0, 10))

    for h in break_ctrlc(tn(z(sampler))):
        # print sampler.chain_idx, h.posterior_score, h
        pass

    for x in tn.get_all(sorted=True):
        print x.posterior_score, x

    print z

    print sampler.nup, sampler.ndown
    print sampler.get_hist()