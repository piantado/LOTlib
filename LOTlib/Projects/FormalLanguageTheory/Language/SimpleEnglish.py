from FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar
from pickle import load

class SimpleEnglish(FormalLanguage):
    """
    A simple English language with a few kinds of recursion all at once
    """
    def __init__(self):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', '%s%s', ['NP', 'VP'], 2.0)
        self.grammar.add_rule('NP', 'd%sn', ['AP'], 1.0)
        self.grammar.add_rule('NP', 'dn', None, 1.0)
        self.grammar.add_rule('NP', 'n', None, 1.0)
        self.grammar.add_rule('AP', 'a%s', ['AP'], 1.0)
        self.grammar.add_rule('AP', 'a',  None, 2.0)

        self.grammar.add_rule('VP', 'v', None, 1.0)
        self.grammar.add_rule('VP', 'v%s', ['NP'], 1.0)
        self.grammar.add_rule('VP', 'vt%s', ['S'], 1.0)
        self.grammar.add_rule('S', 'i%sh%s', ['S','S'], 1.0) # add if S then S grammar

    def terminals(self):
        return list('dnavtih')

# just for testing
if __name__ == '__main__':
    language = SimpleEnglish()
    print language.sample_data(10000)
