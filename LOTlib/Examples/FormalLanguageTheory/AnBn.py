import numpy as np

from LOTlib.Examples.FormalLanguageTheory.FormalLanguage import FormalLanguage, FormalLanguageHypothesis
from LOTlib.Miscellaneous import logsumexp
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str


class AnBn(FormalLanguage):

    def __init__(self, A='a', B='b', max_length=20):
        """
        don't use char like | and ) currently
        """
        self.atom_len = len(A) + len(B)
        assert max_length % self.atom_len == 0, 'max_length should be divisible by len(A)+len(B)'

        FormalLanguage.__init__(self, max_length)

        self.atom = [A, B]
        # assume to be geometric
        self.log_prob_scores = [max_length - k for k in xrange(1, max_length/self.atom_len + 1)]
        self.norm_const = logsumexp(self.log_prob_scores)

    def all_strings(self):
        strings = []

        cnt = self.atom_len
        while cnt <= self.max_length:
            strings.append(self.rep(cnt/self.atom_len))
            cnt += self.atom_len

        return strings

    def is_valid_string(self, s):

        s_len = len(s)
        A_len = len(self.atom[0])
        B_len = len(self.atom[1])
        ind = 0
        A_cnt = 0
        B_cnt = 0

        if s_len % self.atom_len != 0: return False

        while ind < s_len:
            if s[ind:ind+A_len] == self.atom[0]:
                ind += A_len
                A_cnt += 1
            else: break

        while ind < s_len:
            if s[ind:ind+B_len] == self.atom[1]:
                ind += B_len
                B_cnt += 1
            else: break

        if A_cnt == B_cnt and ind == s_len: return True
        return False

    def rep(self, n):
        return ''.join([self.atom[0]*n, self.atom[1]*n])

    def sample_data(self, n, finite=None, avg=True):
        """
        finite: limits the max_length of data
        avg: sample for multiple times and average to reduce noise, note the cnt can have fraction
        """
        length_tmp = self.max_length
        if finite is not None:
            assert isinstance(finite, int) and finite > 0 and finite % self.atom_len == 0, 'invalid input of finite'
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
                    s = self.rep(i+1)
                    if output.has_key(s): output[s] += 1
                    else: output[s] = 1
                    break
        if avg:
            for k in output.keys(): output[k] /= 512.0

        # debug
        print output, n

        self.max_length = length_tmp
        return [FunctionData(input=[], output=output)]

    def string_log_probability(self, s):
        return self.log_prob_scores[len(s)/self.atom_len-1] - self.norm_const


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
    language = AnBn(A='ccc')

    print language.all_strings()
    print language.sample_data(128)

    print language.is_valid_string('ccc')
    print language.is_valid_string('cccb')
    print language.is_valid_string('cccbb')
    print language.is_valid_string('cccccccccb')
    print language.is_valid_string('ccccccbb')