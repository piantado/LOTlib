from utils import *
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Examples.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Examples.FormalLanguageTheory.Language.LongDependency import LongDependency
import time
from pickle import dump

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

    language = LongDependency()

    # show_info('running skewed input case..')
    # rec = probe_MHsampler(make_hypothesis('AnBn'), language.sample_data_as_FuncData, options)
    # dump(rec, open('staged_out' + suffix, 'a'))
    #
    #
    # show_info('running normal input case..')
    # rec1 = probe_MHsampler(make_hypothesis('AnBn'), language.sample_data_as_FuncData, options)
    # dump(rec1, open('normal_out' + suffix, 'a'))