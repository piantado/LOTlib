from utils import *
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Examples.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Examples.FormalLanguageTheory.Language.AnBn import AnBn
import time
from pickle import dump

from LOTlib.DataAndObjects import FunctionData
from collections import Counter

register_primitive(flatten2str)

"""
In this case, we investigate the effect of different observed data distributions on training convergence.
"""

if __name__ == '__main__':
    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================
    (options, args) = parser.parse_args()

    suffix = time.strftime('_' + options.NAME + '_%m%d_%H%M%S', time.localtime())

    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================

    language = AnBn()

    show_info('running skewed input case..')
    rec = probe_MHsampler(make_hypothesis('AnBn'), language.sample_data_as_FuncData, options)
    dump(rec, open('staged_out' + suffix, 'a'))


    show_info('running normal input case..')

    cnt = Counter()
    num = 64.0 * 2 / options.FINITE
    for i in xrange(1, options.FINITE/2+1):
        cnt['a'*i+'b'*i] = num

    rec1 = probe_MHsampler(make_hypothesis('AnBn'), language.sample_data_as_FuncData, options, data=[FunctionData(input=[], output=cnt)])
    dump(rec1, open('normal_out' + suffix, 'a'))