import re

import numpy as np

from LOTlib.Examples.FormalLanguageTheory.FormalLanguage import FormalLanguage, FormalLanguageHypothesis
from LOTlib.Miscellaneous import logsumexp
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str


class Regularlanguage(FormalLanguage):

    def __init__(self, atom='a', max_length=10):
        """
        don't use char like | and ) currently
        """
        FormalLanguage.__init__(self, max_length)
        self.atom = atom
        self.max_length = max_length

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
            return len(s) <= self.max_length
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

        cnt = 0
        prob_sum = 0
        cumu_prob = np.zeros(self.max_length, dtype=np.float64)
        for e in self.all_strings():
            if cnt == 0: prob_sum = self.string_log_probability(e)
            else: prob_sum = logsumexp([prob_sum, self.string_log_probability(e)])
            cumu_prob[cnt] = np.exp(prob_sum)
            cnt += 1
        prob_sum = np.exp(prob_sum)

        output = {}

        if avg: n *= 512
        for _ in xrange(n):
            rand = np.random.rand() * prob_sum
            for i in xrange(self.max_length):
                if rand < cumu_prob[i]:
                    s = self.atom*(i+1)
                    if output.has_key(s): output[s] += 1
                    else: output[s] = 1
                    break
        if avg:
            for k in output.keys(): output[k] /= 512.0

        self.max_length = length_tmp
        return [FunctionData(input=[], output=output)]

    def string_log_probability(self, s):
        return (self.max_length - len(s))/float(len(self.atom))

def make_hypothesis():
    register_primitive(flatten2str)

    TERMINAL_WEIGHT = 2.
    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
    grammar.add_rule('BOOL', 'flip_', [''], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'car_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'recurse_', [], 1.)
    grammar.add_rule('LIST', '[]', None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q('a'), None, TERMINAL_WEIGHT)

    return FormalLanguageHypothesis(grammar)


# just for testing
if __name__ == '__main__':
    language = Regularlanguage()

    print language.all_strings()

    print language.sample_data(128)