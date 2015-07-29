import numpy as np
from collections import Counter
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Miscellaneous import logsumexp
from LOTlib.Hypotheses.Likelihoods.StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis
from LOTlib.Evaluation.EvaluationException import RecursionDepthException


class FormalLanguage(object):
    """
    Set up a class for formal languages, so we can compute things like accuracy, precision, etc.
    """

    def __init__(self):
        pass

    def all_strings(self, max_length=50):
        """ Return all strings up to length maxlength """
        pass

    def string_log_probability(self, s):
        """ the log prob of generating s in this language"""
        return -len(s)

    def is_valid_string(self, s):
        """ Returns True if s is a valid string in this language """
        pass

    def sample_data(self, n, max_length):
        """
        Return a dictionary of {string:count}  that is a sample from this language
        """
        all_strings = list(self.all_strings(max_length=max_length))
        probs = map(self.string_log_probability, all_strings)

        return self.weighted_sample(n, all_strings, probs)

    def sample_data_as_FuncData(self, n, max_length=50, avg=True):
        """
        finite: limits the max_length of data
        avg: sample for multiple times and average to reduce noise, note the cnt can have fraction
        """
        if avg:
            cnt = Counter(self.sample_data(n*512, max_length=max_length))
            n = float(512)
            for key in cnt.keys():
                cnt[key] /= n
            return [FunctionData(input=[], output=cnt)]

        return [FunctionData(input=[], output=Counter(self.sample_data(n, max_length=max_length)))]

    def weighted_sample(self, n, strings, probs):
        length = len(probs)
        prob_sum = logsumexp(probs)
        cumu_prob = np.zeros(length, dtype=np.float64)

        mass = 0
        for i in xrange(length):
            mass += np.exp(probs[i] - prob_sum)
            cumu_prob[i] = mass

        output = []

        for _ in xrange(n):

            rand = np.random.rand()

            for i in xrange(length):
                if rand < cumu_prob[i]:
                    output.append(strings[i])
                    break

        return output

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


class FormalLanguageHypothesis(StochasticFunctionLikelihood, RecursiveLOTHypothesis):
    def __init__(self, grammar=None, **kwargs):
        RecursiveLOTHypothesis.__init__(self, grammar, args=[], recurse_bound=25, maxnodes=100, **kwargs)

    def __call__(self, *args):
        try:
            return RecursiveLOTHypothesis.__call__(self, *args)
        except RecursionDepthException:  # catch recursion and too big
            return None