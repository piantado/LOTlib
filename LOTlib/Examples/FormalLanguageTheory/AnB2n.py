import re
from collections import Counter
from LOTlib.Examples.FormalLanguageTheory.FormalLanguage import FormalLanguage, FormalLanguageHypothesis
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str


class AnB2n(FormalLanguage):

    def __init__(self, A='a', B='bb'):
        """
        don't use char like | and ) currently
        """
        assert len(A) == 1 and len(B) == 2, 'len(A) should be 1 and len(B) should be 2'

        FormalLanguage.__init__(self)

        self.A = A
        self.B = B

    def all_strings(self, max_length=54):

        assert max_length % 3 == 0, 'length should be divisible by 3'

        for i in xrange(1, max_length/3+1):
            yield self.A * i + self.B * i

    def is_valid_string(self, s):
        re_atom = r'%s' % '(' + self.A + '*' + ')' + '(' + self.B + '*' + ')' + '$'
        m = re.match(re_atom, s)
        if m:
            am, bm = m.groups()
            return len(am) * 2 == len(bm)
        else:
            return False

    def string_log_probability(self, s):
        return -len(s)/3


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
    grammar.add_rule('ATOM', q('b'), None, TERMINAL_WEIGHT)

    return FormalLanguageHypothesis(grammar)


# just for testing
if __name__ == '__main__':
    language = AnB2n()

    for e in language.all_strings(max_length=30):
        print e

    print language.sample_data_as_FuncData(128, max_length=30)

    print language.is_valid_string('aaa')
    print language.is_valid_string('abb')
    print language.is_valid_string('abbb')
    print language.is_valid_string('aaabb')
    print language.is_valid_string('aabbbb')