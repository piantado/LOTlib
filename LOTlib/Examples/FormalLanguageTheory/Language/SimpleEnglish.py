from LOTlib.Examples.FormalLanguageTheory.Language.FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar


class SimpleEnglish(FormalLanguage):
    """
    This class of language is generated using following grammar:
        S -> NP VP
        NP -> D A* N
        NP -> P
        VP -> V
        VP -> V NP
        VP -> V T S
    You may want to limit the max_length <= 8, or it can be VERY slow.
    """
    def __init__(self,  D='D', A='A', N='N', P='P', V='V'):
        assert len(D) == len(A) == len(N) == len(P) == len(V), 'length of terminal symbols can only be 1'

        FormalLanguage.__init__(self)

        self.D = D
        self.A = A
        self.N = N
        self.P = P
        self.V = V

        # we use Grammar to help us define this language
        self.grammar = Grammar()
        self.grammar.add_rule('START', ' ', ['NP', 'VP'], 1.0)
        self.grammar.add_rule('NP', ' ', ['D', 'X', 'N'], 1.0)
        self.grammar.add_rule('NP', ' ', ['P'], 1.0)
        self.grammar.add_rule('VP', ' ', ['V'], 1.0)
        self.grammar.add_rule('VP', ' ', ['V', 'NP'], 1.0)
        self.grammar.add_rule('VP', ' ', ['V', '*T', 'START'], 1.0)
        self.grammar.add_rule('X', ' ', ['*A', 'X'], 1.0)

        self.grammar.add_rule('D', '*D', None, 1.0)
        self.grammar.add_rule('X', '*A', None, 1.0)
        self.grammar.add_rule('X', '*', None, 1.0)
        self.grammar.add_rule('N', '*N', None, 1.0)
        self.grammar.add_rule('P', '*P', None, 1.0)
        self.grammar.add_rule('V', '*V', None, 1.0)

        self.str_sets = {}

    def fn_str(self, fn):
        s = str(fn)
        x = ''
        for ind in [i for i in xrange(len(s)) if s.startswith('*', i)]:
            a = s[ind+1:ind+2]
            if a == self.D or a == self.A or a == self.N or a == self.P or a == self.V or a == 'T':
                x += a
        return x

    def all_strings(self, max_length=10):
        i = 2
        flag = True

        while flag:
            flag = False
            for e in self.grammar.enumerate_at_depth(i):
                e = self.fn_str(e)
                e_len = len(e)
                if e_len <= max_length:
                    flag = True
                    self.str_sets[e] = i
                    yield e
            i += 1

    # very slow, maybe write a LR parser?
    def is_valid_string(self, s):
        for e in self.all_strings(len(s)):
            if len(e) == len(s) and e == s:
                return True
        return False

    # let's try this str_prob
    def string_log_probability(self, s):
        if s not in self.str_sets:
            self.all_strings(max_length=len(s))
        return - self.str_sets[s]


# just for testing
if __name__ == '__main__':
    language = SimpleEnglish()

    # for e in language.all_strings(1):
    #     print e

    # for e in language.all_strings(max_length=8):
    #     print e

    print language.sample_data_as_FuncData(300, max_length=5)

    # print language.is_valid_string('PV')
    # print language.is_valid_string('DAANV')
    # print language.is_valid_string('PVTPV')
    # print language.is_valid_string('PVDN')
    # print language.is_valid_string('DNVDDN')