import re
from LOTlib.Examples.FormalLanguageTheory.FormalLanguage import FormalLanguage, FormalLanguageHypothesis
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str


class Regularlanguage(FormalLanguage):

    def __init__(self, atom='a'):
        """
        don't use char like | and ) currently
        """
        FormalLanguage.__init__(self)
        self.atom = atom

    def all_strings(self, max_length=50):

        for i in xrange(1, max_length+1):
            yield self.atom * i

    def is_valid_string(self, s, max_length=50):

        re_atom = r'{}*'.format(self.atom)

        if re.match(re_atom, s):
            return len(s) <= max_length
        else:
            return False


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

    for e in language.all_strings(max_length=10):
        print e

    print language.sample_data_as_FuncData(128, max_length=10)