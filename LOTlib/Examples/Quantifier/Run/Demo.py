# -*- coding: utf-8 -*-
"""
Demo MCMC through lexica. Generally does not work well (too slow) so use the vectorized Gibbs version.

"""
import re
from LOTlib import mh_sample
from LOTlib.Examples.Quantifier.Model import *

if __name__ == "__main__":

    show_baseline_distribution(TESTING_SET)
    print "\n\n"

    # intialize a learner lexicon, at random
    h0 = GriceanQuantifierLexicon(make_my_hypothesis, my_weight_function)

    for w in target.all_words():
        h0.set_word(w) # We will defautly generate from null the grammar if no value is specified

    ### sample the target data
    data = generate_data(300)

    ### Update the target with the data
    target.compute_likelihood(data)

    print h0

    #### Now we have built the data, so run MCMC
    for h in mh_sample(h0, data, 10000000, skip=0):

        sstr = str(h)
        sstr = re.sub("[_ ]", "", sstr)

        sstr = re.sub("presup", u"\u03BB A B . presup", sstr)

        print h.posterior_score, "\t", h.prior, "\t", h.likelihood, "\t", target.likelihood, "\n", sstr, "\n\n"

        #for t in data:
                #print h(t.utterance, t.context), t
