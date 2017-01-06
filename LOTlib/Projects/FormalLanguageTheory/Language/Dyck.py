from FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar

class Dyck(FormalLanguage):

    def __init__(self):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', '(%s)', ['S'], 1.0)
        self.grammar.add_rule('S', '()%s', ['S'], 1.0)
        self.grammar.add_rule('S', '',    None, 1.0)

    def terminals(self):
        return list('()')

# just for testing
if __name__ == '__main__':
    language = Dyck()
    print language.sample_data(10000)