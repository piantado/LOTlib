from collections import Counter
from LOTlib.Examples.FormalLanguageTheory.FormalLanguage import FormalLanguage, FormalLanguageHypothesis
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str


class Dyck(FormalLanguage):
    """
    This one is very hard to learn, please run it with at least 1e5 MCMC steps
    """
    def __init__(self, A='(', B=')'):
        assert len(A) == 1 and len(B) == 1, 'len(A) should be 1 and len(B) should be 1'

        FormalLanguage.__init__(self)

        self.A = A
        self.B = B

    def all_strings(self, max_length=50):
        """
        we iterate over the space using dynamic programming, complexity is O(n^2*L^3), maybe some improvement?
        """
        assert max_length % 2 == 0, 'length should be even'
        memo = Counter([''])
        memo_new = Counter()
        for _ in xrange(max_length/2):
            for e in memo:
                s_len = len(e)
                for i in xrange(s_len+1):
                    for j in xrange(i+1, s_len+2):
                        s = Dyck.insert_str(Dyck.insert_str(e, self.A, i), self.B, j)
                        if self.is_valid_string(s) and not memo_new.has_key(s):
                            memo_new[s] += 1
                            yield s
            memo = memo_new; memo_new = Counter()

    @staticmethod
    def insert_str(s, c, p):
        """
        insert char c into s at position p
        NOTE: not sure it works with negative p !
        """
        s_len = len(s)
        assert 0 <= p <= s_len

        return s[:p] + c + s[p:]

    def is_valid_string(self, s):
        sym_stack = []

        for e in s:
            if e == self.A:
                sym_stack.append(e)
            elif e == self.B:
                if len(sym_stack) > 0:
                    sym_stack.pop(-1)
                else:
                    return False
            else:
                return False

        if len(sym_stack) > 0:
            return False

        return True

    def string_log_probability(self, s):
        return -len(s)/2


def make_hypothesis():
    register_primitive(flatten2str)

    TERMINAL_WEIGHT = 2.
    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
    grammar.add_rule('BOOL', 'flip_', [''], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'car_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'recurse_', [], 1.)
    grammar.add_rule('LIST', '[]', None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q('('), None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q(')'), None, TERMINAL_WEIGHT)

    return FormalLanguageHypothesis(grammar)


# just for testing
if __name__ == '__main__':

    # print Dyck.insert_str('', '(', 0)
    #
    # print Dyck.insert_str('()', '(', 0)
    # print Dyck.insert_str('()', '(', 1)
    # print Dyck.insert_str('()', '(', 2)

    lang = Dyck()

    for e in lang.all_strings(max_length=16):
        print e

    # language = AnBn()
    #
    # for e in language.all_strings(max_length=20):
    #     print e
    #
    # print language.sample_data_as_FuncData(128, max_length=20)
    #
    # print language.is_valid_string('aaa')
    # print language.is_valid_string('ab')
    # print language.is_valid_string('abb')
    # print language.is_valid_string('aaab')
    # print language.is_valid_string('aabb')