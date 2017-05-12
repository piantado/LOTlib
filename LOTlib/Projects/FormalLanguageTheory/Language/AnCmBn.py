import itertools
from LOTlib.Miscellaneous import partitions
from FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar

class AnCmBn(FormalLanguage):

    def __init__(self):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', 'a%sb', ['S'], 1.0)
        self.grammar.add_rule('S', 'a%sb',   ['Cstar'], 1.0)

        self.grammar.add_rule('Cstar', 'c', None, 1.0)
        self.grammar.add_rule('Cstar', 'c%s', ['Cstar'], 1.0)

    def terminals(self):
        return list('abc')

    def all_strings(self):
        for r in itertools.count(1):
            for n,m in partitions(r, 2, 1): # partition into two groups (NOTE: does not return both orders)
                yield 'a'*n + 'c'*m + 'b'*n
                if n != m:
                    yield 'a'*m + 'c'*n + 'b'*m

# just for testing
if __name__ == '__main__':
    language = AnBn()
    print language.sample_data(10000)