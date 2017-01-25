
from FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar

class An(FormalLanguage):

    def __init__(self):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', 'a%s', ['S'], 1.0)
        self.grammar.add_rule('S', 'a',    None, 1.0)

    def terminals(self):
        return list('a')


# just for testing
if __name__ == '__main__':
    language = An()
    print language.sample_data(100)