import re
from LOTlib.Hypotheses.Likelihoods.StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Examples.FormalLanguageTheory.FormalLanguage import FormalLanguage
from LOTlib.Examples.FormalLanguageTheory.Factorized.Hypothesis import FactorizedDataHypothesis, InnerHypothesis
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import q, flatten2str


class AnBnCn(FormalLanguage):

    def __init__(self, A='a', B='b', C='c'):
        """
        don't use char like | and ) currently
        """
        assert len(A) == 1 and len(B) == 1 and len(C) == 1, 'len of A, B, C'

        FormalLanguage.__init__(self)

        self.A = A
        self.B = B
        self.C = C

    def all_strings(self, max_length=50):

        assert max_length % 3 == 0, 'length should be divisible by 3'

        for i in xrange(1, max_length/3+1):
            yield self.A * i + self.B * i + self.C * i

    def is_valid_string(self, s):
        re_atom = r'({}*)({}*)({}*)'.format(self.A, self.B, self.C)

        m = re.match(re_atom, s)
        if m:
            am, bm, cm = m.groups()
            return len(am) == len(bm) == len(cm)
        else:
            return False

    def string_log_probability(self, s):
        return -len(s)/3


def make_hypothesis():
    register_primitive(flatten2str)

    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 0.09)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 0.56)
    grammar.add_rule('BOOL', 'flip_', [''], 0.43)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 0.203)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 0.15)
    grammar.add_rule('LIST', 'car_', ['LIST'], 0.15)
    grammar.add_rule('LIST', '\'\'', None, 0.23)
    grammar.add_rule('ATOM', q('a'), None, .33)
    grammar.add_rule('ATOM', q('b'), None, .33)
    grammar.add_rule('ATOM', q('c'), None, .33)

    return AnBnCnHypothesis(grammar=grammar, N=4, recurse_bound=25, maxnodes=125)


from LOTlib.Miscellaneous import logsumexp
from Levenshtein import distance
from math import log
class AnBnCnHypothesis(StochasticFunctionLikelihood, FactorizedDataHypothesis):
    """
    A particular instantiation of FactorizedDataHypothesis, with a likelihood function based on
    levenshtein distance (with small noise rate -- corresponding to -100*distance)
    """

    def __init__(self, **kwargs):
        FactorizedDataHypothesis.__init__(self, **kwargs)

    def make_hypothesis(self, **kwargs):
        return InnerHypothesis(**kwargs)

    def compute_single_likelihood(self, datum):
        """
            Compute the likelihood with a Levenshtein noise model
        """

        assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"

        llcounts = self.make_ll_counts(datum.input)

        lo = sum(llcounts.values())

        ll = 0.0 # We are going to compute a pseudo-likelihood, counting close strings as being close
        for k in datum.output.keys():
            ll += datum.output[k] * logsumexp([ log(llcounts[r])-log(lo) - 100.0 * distance(r, k) for r in llcounts.keys() ])
        return ll


# just for testing
if __name__ == '__main__':
    language = AnBnCn()

    for e in language.all_strings(max_length=30):
        print e

    print language.sample_data_as_FuncData(128, max_length=30)

    print language.is_valid_string('aaabb')
    print language.is_valid_string('abc')
    print language.is_valid_string('abbcc')
    print language.is_valid_string('aaabc')
    print language.is_valid_string('aabbcc')