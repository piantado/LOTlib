"""

    This likelihood is for stochastic functions. To compute the likelihood, we must simulate forwards a bunch of times.
    (Previously, this was a hypothesis type, SimpleGenerativeHypothesis)

    NOTE: A very subtle error can occur if exceptions (like TooBigException) are caught in __call__, then ll_counts may never get set.
"""

from LOTlib.Miscellaneous import attrmem, nicelog
from collections import Counter

# debug
# from LOTlib.Examples.FormalLanguageTheory.parse_hypothesis import rank

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
import os
from pickle import dump
import numpy as np

total_num = 0

class StochasticFunctionLikelihood(object):

    @attrmem('ll_counts')
    def make_ll_counts(self, input, nsamples=512):
        """
            Run this model forward nsamples times (defaultly self.nsamples),
            returning a dictionary of how often each outcome occurred
        """

        if nsamples is None:
            nsamples = self.nsamples

        llcounts = Counter()

        # f = open('../../Examples/FormalLanguageTheory/ppp_%i' % rank, 'a')
        # if os.stat('../../Examples/FormalLanguageTheory/ppp_%i' % rank).st_size > 1e6:
        #     f.close()
        #     f = open('../../Examples/FormalLanguageTheory/ppp_%i' % rank, 'w')
        save_path = '../../Examples/FormalLanguageTheory/gen/'

        global total_num
        if total_num > 1000:
            for f in os.listdir(save_path): os.remove(save_path+f)
            total_num = 0

        dump(self, open(save_path + '%.8f_%i' % (np.random.rand(), rank), 'w'))
        total_num += 1

        # print >> f, str(self)
        # f.close()
        print str(self)
        for i in xrange(nsamples):
            print i
            llcounts[self(*input)] += 1

        return llcounts


    def compute_single_likelihood(self, datum, llcounts=None, nsamples=512, sm=0.1):
        """
                sm smoothing counts are added to existing bins of counts (just to prevent badness)
                This can take an optiona llcounts in order to allow us to cache this externally
        """
        #print self
        assert isinstance(datum.output, dict), "Data supplied to SimpleGenerativeHypothesis must be a dict (function outputs to counts)"

        if llcounts is None: # compute if not passed in
            llcounts = self.make_ll_counts(datum.input, nsamples=nsamples)

        return sum([ datum.output[k] * (nicelog(llcounts[k] + sm)-nicelog(nsamples + sm*len(datum.output.keys())) ) for k in datum.output.keys() ])

