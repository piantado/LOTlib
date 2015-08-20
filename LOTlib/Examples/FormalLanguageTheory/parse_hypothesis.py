from pickle import load
from collections import Counter

from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
import numpy as np
from Language.SimpleEnglish import SimpleEnglish
register_primitive(flatten2str)

h_set = load(open('hypotheses__0819_183013'))

h_m = None
for h in h_set:
    if 'cons_(\'a\', cons_(\'a\', ( \'\' if empty_(cdr_(recurse_())) else cons_(\'b\', cons_(\'b\', \'\')) ))) if flip_() else' in str(h):
        h_m = h
        print Counter([h() for _ in xrange(1024)])
        print str(h)
#
# DATA_RANGE = np.arange(10, 400, 21)
# language = SimpleEnglish()
#
#
# for size in DATA_RANGE:
#     evaluation_data = language.sample_data_as_FuncData(size, max_length=5)
#     h_m.compute_posterior(evaluation_data)
#     print size, h_m.posterior_score / size