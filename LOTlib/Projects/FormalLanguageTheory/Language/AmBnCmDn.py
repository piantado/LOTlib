import itertools
from LOTlib.Miscellaneous import partitions
from FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar

class AmBnCmDn(FormalLanguage):
    """
    See Shieber 1985
    """

    def __init__(self):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', '%s%s', ['A', 'B'], 1.0)
        self.grammar.add_rule('A', 'a%s',  ['A'], 1.0)
        self.grammar.add_rule('A', 'a',    None, 1.0)
        self.grammar.add_rule('B', 'b%s',  ['B'], 1.0)
        self.grammar.add_rule('B', 'b',    None, 1.0)

    def terminals(self):
        return list('abcd')

    def sample_string(self): # fix that this is not CF
        s = str(self.grammar.generate()) # from a^m b^n
        s = s+'c'*s.count('a') + 'd'*s.count('b')
        return s

    def all_strings(self):
        for r in itertools.count(1):
            for n,m in partitions(r, 2, 1): # partition into two groups (NOTE: does not return both orders)
                yield 'a'*n + 'b'*m + 'c'*n + 'd'*m
                if n != m:
                    yield 'a'*m + 'b'*n + 'c'*m + 'd'*n


# just for testing
if __name__ == '__main__':
    language = AmBnCmDn()
    print language.sample_data(10000)