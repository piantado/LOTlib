from utils import *
from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
from LOTlib.Examples.FormalLanguageTheory.Model.Hypothesis import make_hypothesis
from LOTlib.Examples.FormalLanguageTheory.Language.An import An
import time
from pickle import dump

register_primitive(flatten2str)

if __name__ == '__main__':
    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================
    (options, args) = parser.parse_args()

    suffix = time.strftime('_' + options.NAME + '_%m%d_%H%M%S', time.localtime())
    # ========================================================================================================
    # Process command line arguments /
    # ========================================================================================================

    language = An()

    show_info('running normal input case..')
    sampler = probe_MHsampler(make_hypothesis('An'), language.sample_data_as_FuncData, options, 'without_prior_out' + suffix, ret_sampler=True)

    show_info('running with input using different letter case..')
    language = An(atom='b')
    probe_MHsampler(make_hypothesis('An'), language.sample_data_as_FuncData, options, 'with_prior_out' + suffix, sampler=sampler)