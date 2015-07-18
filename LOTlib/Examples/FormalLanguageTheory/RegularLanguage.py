import re

import numpy as np

from LOTlib.Examples.FormalLanguageTheory.FormalLanguage import FormalLanguage
from LOTlib.Miscellaneous import logsumexp
from LOTlib.DataAndObjects import FunctionData


class Regularlanguage(FormalLanguage):

    def __init__(self, atom='a', max_length=10):
        """
        don't use char like | and ) currently
        """
        FormalLanguage.__init__(self, max_length)
        self.atom = atom

        # assume to be geometric
        self.log_prob_scores = [max_length - k for k in xrange(1, max_length + 1)]
        self.norm_const = logsumexp(self.log_prob_scores)

    def all_strings(self):
        strings = []
        a_len = len(self.atom)

        cnt = a_len
        s = self.atom
        while cnt <= self.max_length:
            strings.append(s)
            s += self.atom
            cnt += a_len

        return strings

    def is_valid_string(self, s):

        re_atom = r'%s' % ('(' + self.atom + ')' + '*')

        if re.match(re_atom, s):
            return True
        else:
            return False

    def sample_data(self, n, finite=None, avg=True):
        """
        finite: limits the max_length of data
        avg: sample for multiple times and average to reduce noise, note the cnt can have fraction
        """
        length_tmp = self.max_length
        if finite is not None:
            assert isinstance(finite, int) and finite > 0, 'invalid input of finite'
            self.max_length = min(self.max_length, finite)

        output = {}

        # get the cumulative probability
        cum_prob = np.zeros(self.max_length, dtype=np.float64)
        prob_sum = 0
        cnt = 0
        for e in self.all_strings():
            prob_sum += np.exp(self.string_log_probability(e))
            cum_prob[cnt] = prob_sum
            cnt += 1

        if avg: n *= 512
        for _ in xrange(n):
            rand = np.random.rand() * prob_sum
            for i in xrange(self.max_length):
                if rand < cum_prob[i]:
                    s = self.atom*(i+1)
                    if output.has_key(s): output[s] += 1
                    else: output[s] = 1
                    break
        if avg:
            for k in output.keys(): output[k] /= 512.0

        self.max_length = length_tmp
        return [FunctionData(input=[], output=output)]

    def string_log_probability(self, s):
        return self.log_prob_scores[len(s)-1] - self.norm_const


# just for testing
if __name__ == '__main__':
    language = Regularlanguage()

    print language.all_strings()

    print language.sample_data(128)