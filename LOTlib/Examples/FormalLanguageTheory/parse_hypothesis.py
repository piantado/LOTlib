from pickle import load
from collections import Counter

from LOTlib.Evaluation.Eval import register_primitive
from LOTlib.Miscellaneous import flatten2str

register_primitive(flatten2str)

h_set = load(open('out//SimpleEnglish//hypotheses_simple_7_20w_0810_002751'))

for h in h_set:
    if 'x0 if flip_() else cons_(\'D\', cons_(\'N\', recurse_(cons_(cdr_(x0), recurse_(cdr_(cdr_(recurse_(recurse_(recurse_(cons_(\'N\', car_(x0))))))))))))' in str(h):
        print Counter([h() for _ in xrange(1024)])
        print str(h)
