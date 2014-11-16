
from collections import defaultdict
from LOTlib.DataAndObjects import UtteranceData
from LOTlib.Miscellaneous import exp
from LOTlib.Examples.Quantifier.Model import *

#distribution of context sizes
#for i in xrange(1000):
        #context = sample_context()

#comparison = GriceanQuantifierLexicon(make_my_hypothesis, my_weight_function)
#comparison.set_word('every', LOTHypothesis(G, value='SET_IN_TARGET', f=lambda A, B, S:  presup_( nonempty_( A ), empty_( A ) )))
#comparison.set_word('some', LOTHypothesis(G, value='SET_IN_TARGET', f=lambda A, B, S: presup_( True, subset_( A, A ) )))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#This will debug -- for a given context and all_utterances, see if our likelihood is the same as empirical sampling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

N_SAMPLES = 10000

if __name__ == "__main__":

    context = sample_context()

    cnt = defaultdict(int)
    for _ in xrange(N_SAMPLES):
        u = target.sample_utterance( context=context, possible_utterances=target.all_words())
        cnt[u] += 1

    #for w,c in cnt.items():
        #print w, float(c)/float(N_SAMPLES), exp(target.compute_single_likelihood( UtteranceData(utterance=w, possible_utterances=target.all_words(), context=context)))

    print check_counts( cnt, lambda w: exp(target.compute_single_likelihood( UtteranceData(utterance=w, possible_utterances=target.all_words(), context=context))))
