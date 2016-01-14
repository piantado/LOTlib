# -*- coding: utf-8 -*-

from LOTlib.Miscellaneous import q, qq, Infinity
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler, MH_acceptance

from math import log, exp
from random import random

class MHSamplerShortcut(MHSampler):
    """A version of MHSampler that uses shortcut evaluation

    """

    def next(self):
        """Generate another sample."""
        if self.samples_yielded >= self.steps:
            raise StopIteration
        else:
            for _ in xrange(self.skip+1):

                self.proposal, fb = self.proposer(self.current_sample)

                assert self.proposal is not self.current_sample, "*** Proposal cannot be the same as the current sample!"
                assert self.proposal.value is not self.current_sample.value, "*** Proposal cannot be the same as the current sample!"

                # compute the shortcut value of the likelihood
                # We will only be accepted if ll < ll_cutoff, which we can use in self.compute_posterior
                # to speed things along
                # Note that this requires passing the same p to MH_acceptance, since it determines the cutoff
                p = random() # the random number
                ll_cutoff = (log(p)*self.acceptance_temperature + \
                            -self.proposal.prior/self.prior_temperature + \
                            self.current_sample.prior/self.prior_temperature + \
                            self.current_sample.likelihood/self.likelihood_temperature + \
                            fb) * self.likelihood_temperature

                # Call myself so memoized subclasses can override
                self.compute_posterior(self.proposal, self.data, shortcut=ll_cutoff)

                # Note: It is important that we re-compute from the temperature since these may be altered
                #    externally from ParallelTempering and others
                prop = (self.proposal.prior/self.prior_temperature +
                        self.proposal.likelihood/self.likelihood_temperature)
                cur = (self.current_sample.prior/self.prior_temperature +
                       self.current_sample.likelihood/self.likelihood_temperature)

                if self.trace:
                    print "# Current: ", round(cur,3), self.current_sample
                    print "# Proposal:", round(prop,3), self.proposal
                    print ""
                
                # if MH_acceptance(cur, prop, fb, acceptance_temperature=self.acceptance_temperature): # this was the old form
                if MH_acceptance(cur, prop, fb, p=p, acceptance_temperature=self.acceptance_temperature):
                    self.current_sample = self.proposal
                    self.was_accepted = True
                    self.acceptance_count += 1
                else:
                    self.was_accepted = False

                self.proposal_count += 1

            self.samples_yielded += 1
            return self.current_sample

if __name__ == "__main__":

    # Just an example
    from LOTlib import break_ctrlc
    from LOTlib.Examples.Number.Model import make_data, NumberExpression, grammar

    data = make_data(300)
    h0 = NumberExpression(grammar)
    sampler = MHSamplerShortcut(h0, data, steps=100000)
    for h in break_ctrlc(sampler):
        print h.posterior_score, h.prior, h.likelihood, h.compute_likelihood(data), h


