from FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar

class AnBnCn(FormalLanguage):

    def __init__(self):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', 'a%sb', ['S'], 1.0)
        self.grammar.add_rule('S', '',    None, 1.0)

    def terminals(self):
        return list('abc')

    def sample_string(self): # fix that this is not CF
        s = str(self.grammar.generate())
        return s + 'c'*(len(s)/2)

# just for testing
if __name__ == '__main__':
    language = AnBnCn()
    print language.sample_data(10000)