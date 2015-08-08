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
    rec, sampler = probe_MHsampler(make_hypothesis('An'), language.sample_data_as_FuncData, options, ret_sampler=True)
    dump(rec, open('staged_out' + suffix, 'a'))


    show_info('running with input using different letter case..')
    language = An(atom='b')
    rec1 = probe_MHsampler(make_hypothesis('An'), language.sample_data_as_FuncData, options, sampler=sampler)
    dump(rec1, open('normal_out' + suffix, 'a'))