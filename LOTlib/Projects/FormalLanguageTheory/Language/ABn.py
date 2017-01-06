
from LOTlib.Projects.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar

class ABn(FormalLanguage):

    def __init__(self):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', 'ab%s', ['S'], 1.0)
        self.grammar.add_rule('S', 'ab', None, 1.0)

    def terminals(self):
        return list('ab')


        # just for testing


if __name__ == '__main__':
    language = ABn()
    print language.sample_data(10000)