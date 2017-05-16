
from FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar

class Saffran(FormalLanguage):
    """
    From Saffran, Aslin, Newport studies.
    Strings consisting of               tupiro golabu bidaku padoti
    coded here with single characters:  tpr     glb    Bdk    PDT
    """
    def __init__(self):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', '%s%s', ['T', 'S'], 1.0)
        self.grammar.add_rule('S', '%s',   ['T'], 1.0)
        self.grammar.add_rule('T', 'tpr',    None, 0.25)
        self.grammar.add_rule('T', 'glb',    None, 0.25)
        self.grammar.add_rule('T', 'Bdk',    None, 0.25)
        self.grammar.add_rule('T', 'PDT',    None, 0.25)

    def terminals(self):
        return list('tprglbBdkPDT')

    def all_strings(self):
        for g in self.grammar.enumerate():
            yield str(g)



# just for testing
if __name__ == '__main__':
    language = Saffran()
    print language.sample_data(10000)

