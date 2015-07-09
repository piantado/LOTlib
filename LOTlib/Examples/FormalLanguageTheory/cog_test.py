from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis
from LOTlib.Hypotheses.Likelihoods.StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Miscellaneous import q, flatten2str
import numpy as np
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Evaluation.EvaluationException import RecursionDepthException
from sys import argv


class MyHypothesis(StochasticFunctionLikelihood, RecursiveLOTHypothesis):
    def __init__(self, grammar=None, **kwargs):
        RecursiveLOTHypothesis.__init__(self, grammar, args=[], prior_temperature=0.2, recurse_bound=25, maxnodes=100, **kwargs)

    def __call__(self, *args):
        try:
            return RecursiveLOTHypothesis.__call__(self, *args)
        except RecursionDepthException:  # catch recursion and too big
            return None


def make_hypothesis():
    register_primitive(flatten2str)

    TERMINAL_WEIGHT = 2.
    grammar = Grammar()
    grammar.add_rule('START', 'flatten2str', ['LIST', 'sep=\"\"'], 1.0)
    grammar.add_rule('BOOL', 'empty_', ['LIST'], 1.)
    grammar.add_rule('BOOL', 'flip_', [''], 1.0)
    grammar.add_rule('LIST', 'if_', ['BOOL', 'LIST', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cons_', ['ATOM', 'LIST'], 1.)
    grammar.add_rule('LIST', 'cdr_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'car_', ['LIST'], 1.)
    grammar.add_rule('LIST', 'recurse_', [], 1.)
    grammar.add_rule('LIST', '[]', None, TERMINAL_WEIGHT)
    grammar.add_rule('ATOM', q('a'), None, TERMINAL_WEIGHT)

    return MyHypothesis(grammar)


def make_data(n=7):
    """
    'a' will be seen for 2^n times; and 'aa' for 2^(n-1). if (n-k) < 0, it stops to generate
    """
    assert n >= 0, 'invalid input %i' % n

    output = {}
    cnt = 1
    while n >= 0:
        output['a'*cnt] = 2**n
        # output['a' * cnt] = 20
        n -= 1
        cnt += 1

    return [FunctionData(input=[], output=output)]


def gen_rotate_data(frame, frame_size=10):
    """
    assume n=7
    for dec: 'a' will be seen for 2^7 times; and 'aa' for 2^6
    for asc: 'a' will be seen for 2^0 times; and 'aa' for 2^1
    we generate data between these two states according to frame/frame_size
    """
    n = 7
    dec = [('a' * cnt, 2 ** (n + 1 - cnt)) for cnt in xrange(1, n+2)]
    asc = [('a' * cnt, 2 ** (cnt - 1)) for cnt in xrange(1, n+2)]

    output = {}
    for i in xrange(len(dec)):
        output[asc[i][0]] = dec[i][1] + frame/float(frame_size) * (asc[i][1] - dec[i][1])

    return [FunctionData(input=[], output=output)]


def gen_rotate_data_zoom(frame, frame_size=30):
    """
    assume n=7
    for dec: 'a' will be seen for 2^7 times; and 'aa' for 2^6
    for asc: 'a' will be seen for 2^0 times; and 'aa' for 2^1
    then the asc is substitue with state frame/frame_size=4/10, so we have a zoom-in effect
    """
    n = 7
    dec = [['a' * cnt, 2 ** (n + 1 - cnt)] for cnt in xrange(1, n+2)]
    asc = [['a' * cnt, 2 ** (cnt - 1)] for cnt in xrange(1, n+2)]
    for i in xrange(len(dec)):
        asc[i][1] = dec[i][1] + 0.4 * (asc[i][1] - dec[i][1])

    output = {}
    for i in xrange(len(dec)):
        output[asc[i][0]] = dec[i][1] + frame/float(frame_size) * (asc[i][1] - dec[i][1])

    return [FunctionData(input=[], output=output)]


def init():
    return make_hypothesis(), make_data()


def is_valid(num):
    return num != float('NaN') and num != float('Inf') and num != -float('Inf')


if __name__ == '__main__':
    # input with num_of_samples and num_of_burn-in
    assert len(argv) == 3, 'Try again'
    num = int(argv[1])
    burn_in = int(argv[2])

    hypo, data = init()

    h_old = hypo.propose()
    for i in xrange(num):
        h = h_old[0].propose()
        acc = h[0].compute_posterior(data) - h_old[0].compute_posterior(data) + h[1]
        if not is_valid(acc): continue

        if acc > 0 or np.random.rand() < np.exp(acc):
            h_old = h

        if i >= burn_in and i % 10 == 0:
            print str(h_old)
            print h_old[0].posterior_score



