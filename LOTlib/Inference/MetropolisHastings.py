# -*- coding: utf-8 -*-
import LOTlib
from LOTlib.Miscellaneous import q, qq, Infinity
from MHShared import MH_acceptance

class MHSampler():
    """
            A class to implement MH sampling. You can create a sampler object::

                    from LOTlib.Examples.Number.Shared import generate_data, NumberExpression, grammar
                    data = generate_data(500)
                    h0 = NumberExpression(grammar)
                    sampler = MHSampler(h0, data, 10000)
                    for h in sampler:
                            print sampler.acceptance_ratio(), h

            Or implicitly::
                    from LOTlib.Examples.Number.Shared import generate_data, NumberExpression, grammar
                    data = generate_data(500)
                    h0 = NumberExpression(grammar)
                    for h in  MHSampler(h0, data, 10000):
                            print h

            A wrapper below called mh_sample is provided to maintain backward compatibility. But mh_sample will be removed in the future.

            If a proposer is specific in __init__, it should return a *new copy* of the object
    """
    def __init__(self, current_sample, data, steps=Infinity, proposer=None, skip=0, prior_temperature=1.0, likelihood_temperature=1.0, acceptance_temperature=1.0, trace=False):
        """
            *current_sample* -- if None, we don't compute it's posterior (via set_state); otherwise we do. 
        """
        self.__dict__.update(locals())

        # was the last proposal accepted
        self.was_accepted = None

        # Defaultly call the hypothesis's propose function
        if proposer is None:
            self.proposer = lambda x: x.propose()

        # how many samples have I yielded? This doesn't count skipped samples
        self.samples_yielded = 0
        
        self.set_state(current_sample, compute_posterior = (current_sample is not None))

        self.reset_counters()

    def set_state(self, x, compute_posterior=True):
        """
            Set the current state, computing the posterior if needed
        """
        
        self.current_sample = x
        
        if compute_posterior:
            self.current_sample.compute_posterior(self.data)
            
        
    def reset_counters(self):
        """
                Reset our acceptance and proposal counters
        """
        self.acceptance_count = 0
        self.proposal_count = 0

    def acceptance_ratio(self):
        """
                Returns the proportion of proposals that have been accepted
        """
        return float(self.acceptance_count) / float(self.proposal_count)

    def take(self, n):
        """
                Yield just the next N from this generator
        """
        raise NotImplementedError

    def compute_posterior(self, h, data):
        """
                A wrapper for hypothesis.compute_posterior(data) that can be overwritten in subclassses for fanciness
                Should return [np,nl], the prior and likelihood
        """
        return h.compute_posterior(data)

    def __iter__(self):
        return self

    def internal_sample(self, h):
        """
                This is called on each yielded h. It serves no function in MHSampler, but is necessary in others
                like TabooMCMC
        """
        pass

    def next(self):
        if self.samples_yielded >= self.steps:
            raise StopIteration
        else:
            for _ in xrange(self.skip+1):

                self.proposal, fb = self.proposer(self.current_sample)

                # either compute this, or use the memoized version
                np, nl = self.compute_posterior(self.proposal, self.data)

                #print np, nl, current_sample.prior, current_sample.likelihood
                # NOTE: IT is important that we re-compute from the temperature since these may be altered externally from ParallelTempering and others
                prop = (np/self.prior_temperature+nl/self.likelihood_temperature)
                cur  = (self.current_sample.prior/self.prior_temperature + self.current_sample.likelihood/self.likelihood_temperature)
                
                #print "# Current:", self.current_sample
                #print "# Proposal:", self.proposal
                
                if MH_acceptance(cur, prop, fb, acceptance_temperature=self.acceptance_temperature):
                    self.current_sample = self.proposal
                    self.was_accepted = True
                    self.acceptance_count += 1
                else:
                    self.was_accepted = False

                self.internal_sample(self.current_sample)
                self.proposal_count += 1

            if self.trace:
                print self.current_sample.posterior_score, self.current_sample.likelihood, self.current_sample.prior, qq(self.current_sample)

            self.samples_yielded += 1
            return self.current_sample

class mh_sample(MHSampler):
    """
            Just temporarily, this will prevent code from breaking, from the switch to MHSampler
    """
    pass


if __name__ == "__main__":

    from LOTlib.Examples.Number.Global import generate_data, NumberExpression, grammar, get_knower_pattern

    data = generate_data(300)
    h0 = NumberExpression(grammar)
    sampler = MHSampler(h0, data, steps=2000)
    for h in sampler:
        print h.posterior_score, h
        #print q(get_knower_pattern(h)), h.posterior_score, h.prior, h.likelihood, q(h), sampler.acceptance_count, sampler.acceptance_ratio()



#       from LOTlib.Examples.Number.Shared import generate_data, NumberExpression, grammar, get_knower_pattern
#
#       data = generate_data(500)
#       h0 = NumberExpression(grammar)
#       for h in mh_sample(h0, data, 10000):
#               print q(get_knower_pattern(h)), h.lp, h.prior, h.likelihood, q(h)
#
#
