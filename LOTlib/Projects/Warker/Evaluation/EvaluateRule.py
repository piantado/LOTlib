from __future__ import division
import pickle
from scipy.misc import logsumexp
import numpy as np
from itertools import product

from LOTlib.DataAndObjects import FunctionData
from LOTlib.Miscellaneous import Infinity, nicelog
from LOTlib.Projects.Warker.Model import MyHypothesis
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

files = [("EnglishWeighted.pkl","englishweighted.csv"), ("FirstSegment.pkl","segment.csv"), ("FirstOrderClassesAllData.pkl","classes.csv"), ("EnglishUniform.pkl","englishuniform.csv")]


for f in files:
    print f[0]
    print f[1]
    print "Loading the hypothesis space . . ."

    spaceset = pickle.load(open(f[0], "r"))

    space = list(spaceset)
    # filter the space for this weird error
    space = [h for h in space if 'll_counts' in dir(h)]
    target = ['h e s', 'm e s', 'm e g', 'h e g', 'm e m', 'm e n', 'h e m', 'm e k', 'k e s', 'h e k', 'k e N', 'k e g', 'h e n', 'k e k', 'm e N', 'k e n', 'h e N', 'f e N', 'g e N', 'n e N', 'n e s', 'f e n', 'g e n', 'g e m', 'f e m', 'g e k', 'f e k', 'g e g', 'f e g', 'f e s', 'n e g', 'k e m', 'n e n', 'n e m', 'g e s', 'n e k']



    training = ['h e s', 'm e s', 'm e g', 'h e g', 'm e n', 'h e m', 'm e k', 'k e s', 'h e k', 'k e N', 'k e g', 'h e n', 'm e N', 'k e n', 'h e N', 'f e N', 'g e N', 'n e N', 'n e s', 'f e n', 'g e n', 'g e m', 'f e m', 'g e k', 'f e k', 'f e g', 'f e s', 'n e g', 'k e m', 'n e m', 'g e s', 'n e k']



    worst = min(space, key=attrgetter('posterior_score'))
    best = max(space, key=attrgetter('posterior_score')) #best posterior
    print("The best hypothesis is: ")
    print best.value
    print best.ll_counts #what are the probabilities for each "word"


    #the data given to the models
    def make_data(size):
            return [FunctionData(input=[],
                                 output={'h e s': size, 'm e s': size, 'm e g': size, 'h e g': size, 'm e n': size, 'h e m': size, 'm e k': size, 'k e s': size, 'h e k': size, 'k e N': size, 'k e g': size, 'h e n': size, 'm e N': size, 'k e n': size, 'h e N': size, 'f e N': size, 'g e N': size, 'n e N': size, 'n e s': size, 'f e n': size, 'g e n': size, 'g e m': size, 'f e m': size, 'g e k': size, 'f e k': size, 'f e g': size, 'f e s': size, 'n e g': size, 'k e m': size, 'n e m': size, 'g e s': size, 'n e k': size})]

    mdata = make_data(1000)
    for h in space:
        print h
        h.likelihood = h.likelihood/sum(mdata[0].output.values())
    with open('/home/Jenna/Desktop/Warker/'+str(f[1]), 'w') as f:
        for damt in xrange(0,1000):
                #weight the posterior by data
                posterior_score = [h.prior + h.likelihood * damt for h in space]
                print "Starting analysis for: " + str(damt) + " data points. Ughhhhh/Yay?"

                f1_target = 0.
                f1_training = 0.
                recall_target=0.
                precision_target =0.
                recall_training = 0.
                precision_training=0.


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


                    p = np.exp((h.prior + h.likelihood * damt)-pdata)
                    trr = float(len(set(h.ll_counts.keys()) & set(training))) / len(training)
                    trp = float(len(set(h.ll_counts.keys()) & set(training))) / len(set(h.ll_counts.keys()))

                    if not (trr + trp ==0):
                        f1 = float(2*(trr*trp))/(trr+trp)
                    else:
                        f1 = 0

                    f1_training += p * f1
                    recall_training += trr * p
                    precision_training += trp * p



                print>>f,f1_target,recall_target,precision_target,f1_training, recall_training, precision_training, damt












