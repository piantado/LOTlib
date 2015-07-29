import re
from LOTlib.Examples.FormalLanguageTheory.FormalLanguage import FormalLanguage, FormalLanguageHypothesis
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str


class AnCstarBn(FormalLanguage):

    def __init__(self, A='a', B='b', C='c'):
        """
        don't use char like | and ) currently
        """
        assert len(A) == 1 and len(B) == 1 and len(C) == 1, 'len of A, B and C should be 1'

        FormalLanguage.__init__(self)

        self.A = A
        self.B = B
        self.C = C

    def all_strings(self, max_length=50):

        assert max_length % 2 == 0, 'length should be even'

        for i in xrange(1, max_length/2+1):
            for j in xrange(max_length - 2*i+1):
                yield self.A * i + self.C * j + self.B * i

    def is_valid_string(self, s):
        re_atom = r'%s' % '(' + self.A + '*' + ')' + '(' + self.C + '*' + ')' + '(' + self.B + '*' + ')'

        m = re.match(re_atom, s)
        if m:
            am, cm, bm = m.groups()
            return len(am) == len(bm)
        else:
            return False


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
    grammar.add_rule('ATOM', q('c'), None, TERMINAL_WEIGHT)

    return FormalLanguageHypothesis(grammar)


# just for testing
if __name__ == '__main__':
    language = AnCstarBn()

    for e in language.all_strings(max_length=20):
        print e

    print language.sample_data_as_FuncData(128, max_length=20)

    print language.is_valid_string('aaac')
    print language.is_valid_string('acb')
    print language.is_valid_string('accbb')
    print language.is_valid_string('aaaccb')
    print language.is_valid_string('aaccbb')