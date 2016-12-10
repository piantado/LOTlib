
from FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar

class AnBn(FormalLanguage):

    def __init__(self):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', 'a%sb', ['S'], 1.0)
        self.grammar.add_rule('S', 'a%sb',   ['Cstar'], 1.0)

        self.grammar.add_rule('Cstar', '', None, 1.0)
        self.grammar.add_rule('Cstar', 'c%s', ['Cstar'], 1.0)

    def terminals(self):
        return list('abc')


# just for testing
if __name__ == '__main__':
    language = AnBn()
    print language.sample_data(10000)