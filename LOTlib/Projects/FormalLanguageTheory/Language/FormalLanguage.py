import numpy as np
from collections import Counter
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Miscellaneous import logsumexp, Infinity, weighted_sample
from copy import deepcopy

class FormalLanguage(object):
    """
    Set up a super-class for formal languages, so we can compute things like accuracy, precision, etc.
    """

    def sample_data(self, n):
        # Sample a string of data
        cnt = Counter()
        for _ in xrange(n):
            s = str(self.grammar.generate())
            cnt[s] += 1

        return [FunctionData(input=(), output=cnt)]

    def terminals(self):
        """ This returns a list of terminal symbols, specific to each language
        """
        raise NotImplementedError

    def estimate_precision_and_recall(self, h, data, truncate=True):
        """
            the precision and recall of h given a data set, data should be usually large

            truncate: ignore those generated strings from h whose lengths exceed the max length in data set, it should
             be set False in finite vs infinite case.
        """
        output = set(data[0].output.keys())
        tmp_list = []
        for _ in xrange(int(sum(data[0].output.values()))):
            try:
                tmp_list.append(h())
            except:
                tmp_list.append('')
        h_llcounts = Counter(tmp_list)
        h_out = set(h_llcounts.keys())

        if truncate:
            max_len = max(map(len, output))
            tmp = deepcopy(h_out)
            for _str in tmp:
                if len(_str) > max_len:
                    h_out.remove(_str)
                    del h_llcounts[_str]

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

        return precision, recall, h_llcounts



    def estimate_KL_divergence(self, h, n=1024, max_length=50):
        """ Estimate the KL divergence between me and h """

        # h_out = Counter([h() for _ in xrange(n)])
        # expect = 0
        # for e in self.all_strings(max_length=max_length):
        #     p = 0
        #     if e in h_out: p = h_out[e] / float(n)
        #         expect +=