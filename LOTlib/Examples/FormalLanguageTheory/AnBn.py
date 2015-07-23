import re
from collections import Counter
from LOTlib.Examples.FormalLanguageTheory.FormalLanguage import FormalLanguage, FormalLanguageHypothesis
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str


class AnBn(FormalLanguage):

    def __init__(self, A='a', B='b'):
        """
        don't use char like | and ) currently
        """
        assert len(A) == 1 and len(B) == 1, 'max_length should be divisible by len(A)+len(B)'

        FormalLanguage.__init__(self)

        self.A = A
        self.B = B

    def all_strings(self, max_length=50):

        assert max_length % 2 == 0, 'length should be even'

        for i in xrange(1, max_length/2+1):
            yield self.A * i + self.B * i

    def is_valid_string(self, s):

        m = re.match(r"(a*)(b*)", s)
        if m:
            am, bm = m.groups()
            return len(am) == len(bm)
        else:
            return False

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

    def string_log_probability(self, s):
        return -len(s)/2


def make_hypothesis():
    register_primitive(flatten2str)

    # TODO not be able to learn a^n b^n. Should modify

    TERMINAL_WEIGHT = 2.
    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
    grammar.add_rule('BOOL', 'flip_', [''], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'car_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'recurse_', [], 1.)
    grammar.add_rule('LIST', '[]', None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q('a'), None, TERMINAL_WEIGHT)

    return FormalLanguageHypothesis(grammar)


# just for testing
if __name__ == '__main__':
    language = AnBn(A='a')

    for e in language.all_strings(max_length=20):
        print e

    print language.sample_data_as_FuncData(128, max_length=20)

    print language.is_valid_string('aaa')
    print language.is_valid_string('ab')
    print language.is_valid_string('abb')
    print language.is_valid_string('aaab')
    print language.is_valid_string('aabb')