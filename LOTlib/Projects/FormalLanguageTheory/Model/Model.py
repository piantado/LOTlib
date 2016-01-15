"""
    A version where we try generating some richer code
"""
from LOTlib.Miscellaneous import q
from LOTlib.Grammar import Grammar

grammar = Grammar(start='PROGRAM')

grammar.add_rule('PROGRAM', '', ['STATEMENT'], 1.0 )
grammar.add_rule('PROGRAM', '%s\n    %s', ['PROGRAM', 'STATEMENT'], 1.0 )

# Or we can introduce a variable, and keep it around for later
# in the program (e.g. the bv cannot be introduced in a statement)
grammar.add_rule('PROGRAM', "<BV>=''\n    %s\n    %s", ['PROGRAM', 'STATEMENT'], 1.0, bv_type='VARIABLE')
grammar.add_rule('PROGRAM', "<BV>=lambda *args:''\n    %s\n    %s", ['PROGRAM', 'STATEMENT'], 1.0, bv_type='FVARIABLE')

# Declare variables
grammar.add_rule('STATEMENT', '%s = %s', ['VARIABLE', 'EXPRESSION'], 1.0)
grammar.add_rule('STATEMENT', '%s = %s', ['FVARIABLE', 'FUNCTION'], 1.0)

grammar.add_rule('FVARIABLE', 'recurse_', None, 1.0) # the built-in name of our recursion
grammar.add_rule('VARIABLE',  'x', None, 1.0) # the built-in name of our argument

grammar.add_rule('FUNCTION', 'lambda', ['EXPRESSION'], 1.0, bv_type='VARIABLE') # just give it a default value (in case of no lambdas)

grammar.add_rule('EXPRESSION', '', ['VARIABLE'], 5.0)
for t in 'a':
    grammar.add_rule('EXPRESSION', "'%s'"%t, None, 5.0)

grammar.add_rule('EXPRESSION', "%s(%s)", ['FVARIABLE', 'EXPRESSION'], 5.0)

grammar.add_rule('EXPRESSION', '(%s+%s)', ['EXPRESSION', 'EXPRESSION'], 1.0)
grammar.add_rule('EXPRESSION', '(%s[0])', ['EXPRESSION'], 1.0)
grammar.add_rule('EXPRESSION', '(%s[1:])', ['EXPRESSION'], 1.0)
grammar.add_rule('EXPRESSION', '(%s if %s else %s)', ['EXPRESSION', 'BOOL', 'EXPRESSION'], 1.0)

grammar.add_rule('BOOL', 'empty_', ['EXPRESSION'], 1.)
grammar.add_rule('BOOL', 'flip_', [''], 1.)

# Make a string for "compiling" hypotheses
# Note this includes a default initialization for y, and allowes G to call itself it if wants
display_string = """
def G(recurse_, x):
    %s
    return x
"""
# Or to a BV, above

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis, RecursionDepthException
from LOTlib.Hypotheses.Likelihoods.StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Miscellaneous import logsumexp
from Levenshtein import distance
from math import log


class LanguageHypothesis(RecursiveLOTHypothesis, StochasticFunctionLikelihood):
    """
        reuse_cnts: Set True if you want to reuse the ll_counts from previous call of make_ll_counts. You may want to
            make sure the previous one is not empty set before you make it fixed.
    """

    def __init__(self, **kwargs):
        RecursiveLOTHypothesis.__init__(self, grammar=grammar, max_nodes=100, **kwargs)
        self.reuse_cnts = False

    def __str__(self):
        return display_string % str(self.value)

    def compile_function(self):
        # Override the default compilation to define g

        exec str(self) in locals() # define a local function called G

        return G # G takes a "recurse" argument that is automatically included through RecursiveLOTHypothesis

    def make_ll_counts(self, input, nsamples=512):
        """
            We'll catch the errors here and return empty counter, so that we don't eval many times with errors
        """
        try:
            self.ll_counts = StochasticFunctionLikelihood.make_ll_counts(self, input, nsamples=nsamples)
        except (NameError, IndexError, RuntimeError, RecursionDepthException): # catch y3=y3 kinds of errors, or ''[0], or infinite recursion -- ugh!
            self.ll_counts = {'': nsamples} # If any error, give back nothing

    def compute_single_likelihood(self, datum):
        """
            Compute the likelihood with a Levenshtein noise model
        """
        assert isinstance(datum.output, dict), "Data supplied must be a dict (function outputs to counts)"

        if not self.reuse_cnts: self.make_ll_counts(datum.input) # assign to self.ll_counts

        lo = sum(self.ll_counts.values())

        ll = 0.0 # We are going to compute a pseudo-likelihood, counting close strings as being close
        for k in datum.output.keys():
            ll += datum.output[k] * logsumexp([ log(self.ll_counts[r])-log(lo) - 100.0 * distance(r, k) for r in self.ll_counts.keys() ])
        return ll


def make_hypothesis(**kwargs):
    """
        NOTE: grammar only has atom a, you need to add other atoms yourself
    """
    if options.LANG == 'SimpleEnglish':
        for e in 'dvtn':
                grammar.add_rule('ATOM', q(e), None, 2)

    if 'terminals' in kwargs:
        terminals = kwargs.pop('terminals')
        if terminals is not None:
            for e in terminals:
                grammar.add_rule('ATOM', q(e), None, 2)

    return LanguageHypothesis(**kwargs)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import LOTlib
from LOTlib.Inference.Samplers.StandardSample import standard_sample
import time
import sys
import codecs
from LOTlib.Miscellaneous import display_option_summary
import numpy as np
from LOTlib.Projects.FormalLanguageTheory.Language.Index import instance
import itertools
from pickle import dump, load

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
fff = sys.stdout.flush

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--language", dest="LANG", type="string", default='An', help="name of a language")
parser.add_option("--steps", dest="STEPS", type="int", default=40000, help="Number of samples to run")
parser.add_option("--top", dest="TOP_COUNT", type="int", default=10, help="Top number of hypotheses to store")
parser.add_option("--finite", dest="FINITE", type="int", default=10, help="specify the max_length to make language finite")
parser.add_option("--terminal", dest="TERMINALS", type="string", default='', help="extra terminals")
parser.add_option("--bound", dest="BOUND", type="int", default=50, help="recursion bound")
(options, args) = parser.parse_args()


def run(mk_hypothesis, lang, finite):
    """
    This out on the DATA_RANGE amounts of data and returns all hypotheses in top count
    """
    if LOTlib.SIG_INTERRUPTED:
        return set()

    return standard_sample(lambda: mk_hypothesis(terminals=options.TERMINALS, bound=options.BOUND),
                           lambda: lang.sample_data_as_FuncData(finite),
                           N=options.TOP_COUNT,
                           steps=options.STEPS,
                           show=True, show_skip=200, skip=10, save_top=None)


def simple_mpi_map(run, args):
    print 'rank: ', rank, 'running..'; fff()
    hypo_set = run(*(args[rank]))

    dump(hypo_set, open(prefix+'hypotheses_'+options.LANG+'_%i'%rank+suffix, 'w'))


if __name__ == '__main__':
    """
        example script:
            mpiexec -n 24 python Model.py --language=An --finite=10
            mpiexec -n 24 python Model.py --language=AnBn --finite=20 --terminal=b
            mpiexec -n 24 python Model.py --language=ABn --finite=20 --terminal=b
            mpiexec -n 24 python Model.py --language=AnB2n --finite=30 --terminal=b
            mpiexec -n 24 python Model.py --language=AnBnCn --finite=18 --terminal=bc
            mpiexec -n 24 python Model.py --language=Dyck --finite=8 --terminal=b --steps=100000
            mpiexec -n 24 python Model.py --language=SimpleEnglish --finite=8 --steps=100000
    """
    prefix = 'out/'
    # prefix = '/home/lijm/WORK/yuan/lot/'
    suffix = time.strftime('_%m%d_%H%M%S', time.localtime())

    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    if rank == 0: display_option_summary(options); fff()

    DATA_RANGE = np.arange(0, 70, 3)

    language = instance(options.LANG, options.FINITE)
    args = list(itertools.product([make_hypothesis], [language], DATA_RANGE))

    hypotheses = simple_mpi_map(run, args)






# if __name__=="__main__":
#
#     from LOTlib import break_ctrlc
#     from LOTlib.DataAndObjects import FunctionData
#
#     data = [FunctionData(input=[''], output={'ab': 82.728515625, 'aabb': 30.1796875, 'aaabbb': 11.01953125, 'aaaabbbb': 4.072265625})]
#
#     from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
#     # h0 = LanguageHypothesis()
#     h0 = make_hypothesis(terminals=options.TERMINALS, bound=options.BOUND)
#
#     from LOTlib.TopN import TopN
#     best_hypotheses = TopN(N=10)
#
#     for i, h in enumerate(break_ctrlc(MHSampler(h0, data, skip=10, steps=600))):
#         if i % 50 == 0:
#             print i
#             best_hypotheses.add(h)
#             print h.posterior_score, h.prior, h.likelihood
#             print h
#             print "LL COUNTS ARE:", h.ll_counts
#             print "------------"
#
#     f = open('hypo_%i' % rank, 'w')
#     for h in best_hypotheses:
#         print >> f, h.posterior_score, h.prior, h.likelihood
#         print >> f, h
#         print >> f, "LL COUNTS ARE:", h.ll_counts
#         print >> f, "------------"
#     f.close()