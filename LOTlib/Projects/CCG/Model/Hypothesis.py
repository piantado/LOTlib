from copy import copy
from math import log

from LOTlib.Hypotheses.Lexicon.WeightedLexicon import WeightedLexicon
from LOTlib.DataAndObjects import UtteranceData
from LOTlib.Examples.CCG.Model.Utilities import can_compose
from LOTlib.DataAndObjects import Context

from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis

from Grammar import grammar
from Data import all_words ## NOT ideal here, but just an example

def make_hypothesis():

    h = CCGLexicon(alpha=0.9, palpha=0.9, likelihood_temperature=1.0)
    for w in all_words:
        h.set_word(w, LOTHypothesis(grammar, args=['C']))

    return h

class CCGLexicon(WeightedLexicon):
    """A version for doing CCG, which parses in the likelihood."""

    def can_parse(self, sentence):
        """
        A very quick and dirty backtracking parsing algorithm that uses the types to see if we can parse,
        returning a sentence string, a type, and a function computing the truth value.

        """
        assert not isinstance(sentence, UtteranceData), "can_parse takes a sentence, not utterance data. Maybe you forgot .utterance?"

        def inner_parse(s, t, m): # Sentence, type, meaning
            assert len(s)==len(t)==len(m)

            if len(s) == 1:
                # Only return if we are now a primitive type, BOOL
                if t[0] == 'BOOL': return (s[0],t[0],m[0]) # unlist these
                else:              return False

            # Else we try to collapse each adjacent pair of words, compositionally
            for i in xrange(len(s)-1):
                for x,y in ( (i,i+1), (i+1,i) ): # the possible orders -- this ordering sets the priority of the parsing
                    c = can_compose(t[x], t[y])
                    if c:
                        sprime = copy(s)
                        tprime = copy(t)
                        mprime = copy(m)

                        assert min(x,y) == i and max(x,y) == i+1 # no monkey business

                        sprime[i:i+2] = ["(%s %s)" % (s[x], s[y]),] # keep track of the parse
                        tprime[i:i+2] = [c,] # keep track of the resulting type
                        mprime[i:i+2] = [ lambda C: (m[x](C))(m[y](C)), ] # keep the wrapping for C

                        return inner_parse(sprime, tprime, mprime)
            return False

        return inner_parse(sentence, \
                           map(lambda x: self.value[x].type(), sentence),\
                           map(lambda x: self.value[x], sentence))

    def compute_single_likelihood(self, udi):
        """
        TODO: WE CAN USE LIKELIHOOD FROM WEIGHTEDLEXICON, BUT THAT BEHAVES WEIRDLY WHEN THE
        POSSIBLE UTTERANCES ARE SMALL

        """
        assert isinstance(udi, UtteranceData)

        # Types of utterances
        trues, falses, others = self.partition_utterances( udi.possible_utterances, udi.context)
        u = udi.utterance

        # compute the weights
        all_weights  = sum(map(lambda u: self.weightfunction(u, udi.context), udi.possible_utterances ))
        true_weights = sum(map(lambda u: self.weightfunction(u, udi.context), trues))
        met_weights  = sum(map(lambda u: self.weightfunction(u, udi.context), falses)) + true_weights

        # Unlike WeightedLexicon, this doesn't play nicely with the case where we are generating and
        # sometimes trues or mets are empty
        w = self.weightfunction(u, udi.context) # the current word weight
        if   (u in trues):  p = self.palpha * (self.alpha * w / true_weights + (1.0 - self.alpha) * w / met_weights) + (1.0 - self.palpha) * w / all_weights # choose from the trues
        elif (u in falses): p = self.palpha * (1.0-self.alpha) * w / met_weights + (1.0 - self.palpha) * w / all_weights # choose from the trues
        else:               p = (1.0 - self.palpha) * w / all_weights

        return log(p)


    def __call__(self, utterance, context):
        """Evaluate this lexicon on a possible utterance."""
        ret = self.can_parse(utterance)
        if ret:
            assert len(ret) == 3
            s,t,f = ret
            assert t == 'BOOL' # we must eval to this type

            return f(context)
        return None
