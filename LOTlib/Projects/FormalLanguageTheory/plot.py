"""
    This version incrementally adds symbols and then does not change them.

    Todo: show highest prob "mistake" strings

"""
import sys
import codecs
import itertools
import numpy
import operator
from copy import copy
from LOTlib import break_ctrlc
from pickle import load
from math import log, exp
from copy import deepcopy
import random
import numpy as np
import LOTlib
import LOTlib.Primitives
from Language import *

from LOTlib.Miscellaneous import display_option_summary, logsumexp


if __name__ == "__main__":

    Nstrings = 1000 # gather this many total from each language
    Ntop     = 100 # compute the intersection with this many
    LARGE_SAMPLE = 100000 # sample this many to compute the average ll
    # data_range = np.linspace(0, 50, num=Ndata)

    ## We have to increase the strings in order to generate the large set
    import Model
    Model.MAX_SELF_RECURSION = 1000 # needed for bigger sets of strings


    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--language", dest="LANG", type="string", default='An', help="name of a language")
    parser.add_option("--ndata", dest="NDATA", type="int", default=1000, help="Max number of data points seen")
    (options, args) = parser.parse_args()

    language = eval(options.LANG+"()")
    print "# Loaded language ", options.LANG

    data = language.sample_data(LARGE_SAMPLE)
    # datas = [language.sample_data(i) for i in xrange(Ndata)]
    print "# Sampled data for ", options.LANG

    data_range = xrange(options.NDATA)

    # We need to be a little fancy here because we may enumerate strings in different orders, depending
    # on the details of how flip vs all_strings are implemented. This means we have to check overlap between
    # the top strings with different sized sets

    data_strings = set()
    top_data_strings = set()
    for i, w in enumerate(itertools.islice(language.all_strings(), Nstrings)):
        if len(w) > LOTlib.Primitives.MAX_STRING_LENGTH:
            break
        else:
            data_strings.add(w)

            if i < Ntop: top_data_strings.add(w)
    print "# Computed top data strings for ", options.LANG

    with open("out/"+options.LANG+".pkl", 'rb') as f:
        hypotheses = list(load(f))
    print "# Read hypotheses for ", options.LANG
    assert len(hypotheses)>0, "*** No hypotheses read for %s"%options.LANG

    for h in hypotheses:
        h.compute_posterior(data)
        # h.posteriors = [h.compute_posterior(d) for d in datas]

        # and store the output strings just once
        v = h()
        h.strings = set(v.keys())
        h.top_strings = set()
        for i, w in enumerate(sorted(h.strings, reverse=True, key=lambda x: v[x])[:Nstrings]):
            if i < Ntop: h.top_strings.add(w)
            else:        break

        h.recall    = float(len(h.strings.intersection(top_data_strings))) / len(top_data_strings)
        if len(h.top_strings) > 0:
            h.precision = float(len(h.top_strings.intersection(data_strings))) / len(h.top_strings)
        else:
            h.precision = 0.0

        # print h.prior, h.likelihood
        # print h
        # print sorted(list(h.top_strings))
        # print sorted(list(top_data_strings))
        # print top_data_strings - h.top_strings
        # print "---------------"

    print "# Computed hypotheses for ", options.LANG

    precision, recall = numpy.zeros(options.NDATA), numpy.zeros(options.NDATA)

    for i, di in enumerate(data_range):

        posteriors = [h.prior + h.likelihood * float(di) / float(LARGE_SAMPLE) for h in hypotheses]
        # posteriors = [h.posteriors[i] for h in hypotheses]
        Z = logsumexp(posteriors)

        # print [(h.accuracy, exp(p-Z)) for h,p in zip(hypotheses, posteriors)]

        precision[i] = sum([h.precision*exp(p-Z) for h,p in zip(hypotheses, posteriors)])
        recall[i]    = sum([h.recall   *exp(p-Z) for h,p in zip(hypotheses, posteriors)])
    print "# Computed precision and recall for ", options.LANG, precision[-1], recall[-1]

    ####################################################################################################################
    # Plot it
    ####################################################################################################################

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(2,1.5))
    p = fig.add_subplot(111)
    p.semilogx(data_range, precision, linewidth=3)
    p.semilogx(data_range, recall, linewidth=3, linestyle="dashed",)
    xlabel = p.set_xlabel('Amount of data')
    title  = p.set_title(options.LANG)
    p.set_ylim(0,1.05)
    fig.savefig('plots/%s.pdf'%options.LANG, bbox_extra_artists=[xlabel], bbox_inches='tight')
    # plt.show()



