import numpy as np
from collections import Counter
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Miscellaneous import logsumexp, Infinity, weighted_sample


class FormalLanguage(object):
    """
    Set up a super-class for formal languages, so we can compute things like accuracy, precision, etc.
    """

    def __init__(self, max_length=10):

        # Populate our set
        self.str_sets = []
        for s in self.all_strings(max_length):
            self.str_sets.append(s)

    def all_strings(self, max_length):
        """ Return all strings up to length maxlength """

        raise NotImplementedError

    def string_log_probability(self, s):
        """ the log prob of generating s in this language"""
        if s in self.str_sets:
            return -len(s)
        else:
            return -Infinity

    def is_valid_string(self, s):
        """ Returns True if s is a valid string in this language """
        return s in self.str_sets

    def sample_data(self, n):
        """
        Return a dictionary of {string:count}  that is a sample from this language
        """
        return weighted_sample(self.str_sets, N=n, probs=self.string_log_probability, log=True)

    def sample_data_as_FuncData(self, n, avg=True):
        """
        n: can be float in avg mode
        finite: limits the max_length of data
        avg: sample for multiple times and average to reduce noise, note the cnt can have fraction
        """
        if n == 0:
            return [FunctionData(input=[], output=Counter())]

        if avg:
            cnt = Counter(self.sample_data(int(n*512)))
            n = float(512)
            for key in cnt.keys():
                cnt[key] /= n
            return [FunctionData(input=[], output=cnt)]

        return [FunctionData(input=[], output=Counter(self.sample_data(n)))]

    def estimate_precision_and_recall(self, h, data):
        """
        the precision and recall of h given a data set, it should be usually large
        """
        output = set(data[0].output.keys())
        h_out = set([h() for _ in xrange(int(sum(data[0].output.values())))])

        base = len(h_out)
        cnt = 0.0
        for v in h_out:
            if v in output: cnt += 1
        precision = cnt / base

        base = len(output)
        cnt = 0.0
        for v in output:
            if v in h_out: cnt += 1
        recall = cnt / base

        return precision, recall

    def estimate_KL_divergence(self, h, n=1024, max_length=50):
        """ Estimate the KL divergence between me and h """

        # h_out = Counter([h() for _ in xrange(n)])
        # expect = 0
        # for e in self.all_strings(max_length=max_length):
        #     p = 0
        #     if e in h_out: p = h_out[e] / float(n)
        #         expect +=