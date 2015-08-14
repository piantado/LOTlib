from pickle import load
from collections import Counter

from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str
import numpy as np
from Language.SimpleEnglish import SimpleEnglish
register_primitive(flatten2str)

h_set = load(open('out//SimpleEnglish/hypotheses_simple_5_20w_0813_004408'))

h_m = None
for h in h_set:
    if 'cons_(\'P\', x1),' in str(h) and 'cons_(\'N\', x1),' in str(h) and 'recurse_(x3, x3, cons_(x3, car_(x4)), x4, recurse_(x1, x2, x4, x0, x2' in str(h):
        h_m = h
        # print Counter([h() for _ in xrange(1024)])
        # print str(h)

DATA_RANGE = np.arange(10, 400, 21)
language = SimpleEnglish()


for size in DATA_RANGE:
    evaluation_data = language.sample_data_as_FuncData(size, max_length=5)
    h_m.compute_posterior(evaluation_data)
    print size, h_m.posterior_score / size