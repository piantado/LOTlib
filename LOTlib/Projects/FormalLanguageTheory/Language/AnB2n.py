from LOTlib.Grammar import Grammar
from FormalLanguage import FormalLanguage


class AnB2n(FormalLanguage):

    def __init__(self):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', 'a%sbb', ['S'], 1.0)
        self.grammar.add_rule('S', '',    None, 1.0)

    def terminals(self):
        return list('ab')


# just for testing
if __name__ == '__main__':
    language = AnBn()
    print language.sample_data(10000)