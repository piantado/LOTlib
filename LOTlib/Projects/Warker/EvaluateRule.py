from __future__ import division
import pickle
from scipy.misc import logsumexp
import numpy as np
from itertools import product

from DataAndObjects import FunctionData
from LOTlib.Miscellaneous import Infinity, nicelog
from Projects.Warker.Model import MyHypothesis
from operator import attrgetter
from optparse import OptionParser

'''p = OptionParser()
p.add_option("-s", "--space", dest="spacefile", help="name of pkl file of hypotheses to evaluate")
p.add_option("-o", "--order", dest="order", help="which model? 'first' or 'second'?")
p.add_option("-f", "--file", dest="file", help="output csv file", default="outputcsv.csv")
p.add_option("-c", action="store_true", dest="classes", help="Is this the Classes Model? True or False?",default="False")
p.add_option("-e", action="store_true", dest="english", help="Is this the English Model? True or False?",default="False")

(options, args) = p.parse_args()
print options
print args'''
spacefile="FirstSegment.pkl"
file="WHATISHAPPENING.csv"
english=False
classes=True
order="first"
print "Loading the hypothesis space . . ."
spaceset = pickle.load(open(spacefile, "r"))

space = list(spaceset)
# filter the space
space = [h for h in space if 'll_counts' in dir(h)]
precision = ['h e s', 'm e s', 'm e g', 'h e g', 'm e m', 'm e n', 'h e m', 'm e k', 'k e s', 'h e k', 'k e N', 'k e g', 'h e n', 'k e k', 'm e N', 'k e n', 'h e N', 'f e N', 'g e N', 'n e N', 'n e s', 'f e n', 'g e n', 'g e m', 'f e m', 'g e k', 'f e k', 'g e g', 'f e g', 'f e s', 'n e g', 'k e m', 'n e n', 'n e m', 'g e s', 'n e k']
print(len(precision))


recall = ['h e s', 'm e s', 'm e g', 'h e g', 'm e n', 'h e m', 'm e k', 'k e s', 'h e k', 'k e N', 'k e g', 'h e n', 'm e N', 'k e n', 'h e N', 'f e N', 'g e N', 'n e N', 'n e s', 'f e n', 'g e n', 'g e m', 'f e m', 'g e k', 'f e k', 'f e g', 'f e s', 'n e g', 'k e m', 'n e m', 'g e s', 'n e k']



worst = min(space, key=attrgetter('posterior_score'))
best =max(space, key=attrgetter('posterior_score'))
print("The best hypothesis is: ")
print best.value
print best.ll_counts


#the data given to the models
def make_data(size):
        return [FunctionData(input=[],
                             output={'h e s': size, 'm e s': size, 'm e g': size, 'h e g': size, 'm e n': size, 'h e m': size, 'm e k': size, 'k e s': size, 'h e k': size, 'k e N': size, 'k e g': size, 'h e n': size, 'm e N': size, 'k e n': size, 'h e N': size, 'f e N': size, 'g e N': size, 'n e N': size, 'n e s': size, 'f e n': size, 'g e n': size, 'g e m': size, 'f e m': size, 'g e k': size, 'f e k': size, 'f e g': size, 'f e s': size, 'n e g': size, 'k e m': size, 'n e m': size, 'g e s': size, 'n e k': size})]

mdata = make_data(1000)
with open('/home/Jenna/Desktop/Warker/'+str(file), 'w') as f:
    f.write("PrecisionGotem,Weighted,Data\n")
    for damt in xrange(0,1000,10):
            #weight the posterior by data
            posterior_score = [h.prior + h.likelihood * damt for h in space]
            print "Starting analysis for: " + str(damt) + " data points. Ughhhhh/Yay?"
            #normalizing constant
            pdata = logsumexp(posterior_score)
            for h in space:
                fit = float((sum([h.ll_counts[w] for w in precision if w in h.ll_counts.keys()]))/sum(h.ll_counts.values()))
                weight = float(len({k: h.ll_counts[k] for k in h.ll_counts.viewkeys() & set(precision)})/len(precision))
                likely = np.exp((h.prior + h.likelihood * damt)-pdata)

                pp = (fit*weight) *likely


                f.write(str(fit*weight) + ',')
                f.write(str(pp) + ',')
                f.write(str(damt))
                f.write('\n')












