from LOTlib.Examples.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar

class SimpleEnglish(FormalLanguage):
    """
    A simple English language with a few kinds of recursion all at once
    """
    def __init__(self, max_length=10):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', 'S', ['NP', 'VP'], 1.0)
        self.grammar.add_rule('NP', 'NP', ['d', 'AP', 'n'], 1.0)
        self.grammar.add_rule('AP', 'AP', ['a', 'AP'], 1.0)
        self.grammar.add_rule('AP', 'AP', None, 1.0)

        self.grammar.add_rule('VP', 'VP', ['v'], 1.0)
        self.grammar.add_rule('VP', 'VP', ['v', 'NP'], 1.0)
        self.grammar.add_rule('VP', 'VP', ['v', 't', 'S'], 1.0)

        FormalLanguage.__init__(self) ## calls and stores all_strings

    def all_strings(self, max_length=10):
        for x in self.grammar.enumerate(d=max_length):
            s = ''.join(x.all_leaves())
            if len(s) < max_length:
                yield s


# just for testing
if __name__ == '__main__':
    language = SimpleEnglish()

    # for _ in xrange(100):
    #     x =  language.grammar.generate()
    #     print list(x.all_leaves())

    # for t in language.grammar.enumerate():
    #     print t

    # for e in language.all_strings():
    #     print e

    # for e in language.all_strings(max_length=8):
    #     print e

    print language.sample_data_as_FuncData(300, max_length=5)

    # print language.is_valid_string('PV')
    # print language.is_valid_string('DAANV')
    # print language.is_valid_string('PVTPV')
    # print language.is_valid_string('PVDN')
    # print language.is_valid_string('DNVDDN')