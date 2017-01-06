
from __future__ import division
import pickle
from scipy.misc import logsumexp
import numpy as np
from itertools import product

from LOTlib.DataAndObjects import FunctionData
from LOTlib.Miscellaneous import Infinity, nicelog
from LOTlib.Projects.Warker.Model import MyHypothesis
from operator import attrgetter


files = [("SecondPartition.pkl","secondexponly.csv")]

for f in files:
    print f[0]
    print f[1]
    print "Loading the hypothesis space . . ."

    spaceset = pickle.load(open(f[0], "r"))

    space = list(spaceset)
    # filter the space
    space = [h for h in space if 'll_counts' in dir(h)]





    for h in space:
        print h
        print h.posterior_score

from LOTlib.Projects.Warker.KaggikPartition import make_data, grammar
from LOTlib.Primitives import *
from LOTlib.Miscellaneous import flatten2str
data = make_data(3000)
target = MyHypothesis(grammar)
target.force_function(lambda:flatten2str(if_(flip_(), if_(flip_(), if_(flip_(), sample_("nm"), sample_("m")), if_(flip_(), sample_("a"), sample_("m"))), if_(flip_(), if_(flip_(), sample_("si"), sample_("i")), if_(flip_(), sample_("h"), sample_("m"))))))
target.make_ll_counts(data[0].input, nsamples=10000)
print target.ll_counts


'''
    #the data given to the models
    def make_data(size):
        return [FunctionData(input=[],
                             output={'n i k': size, 'h i N': size, 'f a n': size, 'g i f': size, 'm a N': size, 'n a s': size, 'g i k': size, 'k a n': size, 'n i f': size, 'f a g': size, 'g i m': size, 'g i s': size, 's i f': size, 'k i m': size, 'n i m': size, 'g a s': size, 'k a f': size, 'f a s': size, 's i n': size, 's a f': size, 's i k': size, 's i m': size, 'h i m': size, 'h i n': size, 'f a N': size, 'h i k': size, 'k a m': size, 'h i f': size, 'f a m': size, 'g i N': size, 'm i f': size, 'n i s': size, 'k i N': size, 's i N': size, 'n a m': size, 'h i s': size, 'f i s': size, 'k a s': size, 'g a n': size, 'g a m': size, 'h a f': size, 'k i s': size, 'm i n': size, 'k a N': size, 'g a f': size, 'g i n': size, 'k a g': size, 's a n': size, 's a m': size, 'n a f': size, 'n a g': size, 'm i N': size, 's a g': size, 'f i k': size, 'h a N': size, 'f i n': size, 'f i m': size, 'm a s': size, 'g a N': size, 'h a s': size, 'k i f': size, 'n a N': size, 'm i s': size, 's a N': size, 'm i k': size, 'h a g': size, 'm a g': size, 'm a f': size, 'k i n': size, 'h a m': size, 'h a n': size, 'n i N': size, 'f i N': size, 'm a n': size})]


    mdata = make_data(1000)
    target = mdata[0].output.keys()
    print target
    for h in space:
        print h
        h.likelihood = h.likelihood/sum(mdata[0].output.values())


    with open('/home/Jenna/Desktop/Warker/'+str(f[1]), 'w') as f:
        for damt in xrange(0,1000):
                #weight the posterior by data
                posterior_score = [h.prior + h.likelihood * damt for h in space]
                print "Starting analysis for: " + str(damt) + " data points. Ughhhhh/Yay?"

                f1_target = 0.
                recall_target=0.
                precision_target =0.



                #normalizing constant
                pdata = logsumexp(posterior_score)



                for h in space:
                    p = np.exp((h.prior + h.likelihood * damt)-pdata)
                    tr = float(len(set(h.ll_counts.keys()) & set(target))) / len(target)
                    tp = float(len(set(h.ll_counts.keys()) & set(target))) / len(set(h.ll_counts.keys()))

                    if not (tr + tp == 0):
                        f1 = float(2*(tr*tp))/(tr+tp)
                    else:
                        f1 = 0

                    f1_target += p * f1
                    recall_target += tr * p
                    precision_target += tp * p

                    print("The best hypothesis is: ")
                    print(max(space, key=attrgetter('posterior_score')).value)


                #print>>f,f1_target,recall_target,precision_target, damt'''












