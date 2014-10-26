# -*- coding: utf-8 -*-
"""
        Shared functions for the quantifier learning model.
"""

from LOTlib import lot_iter

from LOTlib.Grammar import Grammar
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.Miscellaneous import *
from LOTlib.DataAndObjects import *
from LOTlib.FunctionNode import FunctionNode
from LOTlib.FiniteBestSet import FiniteBestSet

from random import randint
from copy import copy

from GriceanWeightedLexicon import *
from Utilities import *

############################################################
# Store a context

class MyContext(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)


############################################################
# Making data, sets, and objects

# quantifiers involving cardinality
all_objects = make_all_objects(shape=['man', 'woman', 'child'], job=['pirate', 'chef', 'fireman'])

def sample_context():

    set_size = randint(1,8) # the i'th data point
    #set_size =  weighted_sample( range(1,10+1), probs=[7187, 1484, 593, 334, 297, 165, 151, 86, 105, 112] ) # for the number-style probabilities

    # get the objects in the current set
    si = sample_sets_of_objects(set_size, all_objects)

    return MyContext( A=set([o for o in si if o.shape=='man']),\
                      B=set([o for o in si if o.job=='pirate']),\
                      S=set(si))

def generate_data(data_size):

    all_words = target.all_words()

    data = []
    for i in lot_iter(xrange(data_size)):

        # a context is a set of men, pirates, and everything. functions are applied to this to get truth values
        context = sample_context()

        word = target.sample_utterance(all_words, context)

        data.append( UtteranceData(utterance=word, context=context, possible_utterances=all_words) )

    return data


############################################################
# Set up the grammar
grammar = Grammar()

"""
        Note: This was updated on Dec 3 2012, after the language submission. We now include AND/OR/NOT, and S, and removed nonempty
"""
grammar.add_rule('START', 'presup_', ['BOOL', 'BOOL'], 1.0)

grammar.add_rule('START', 'presup_', ['True', 'BOOL'], 1.0)
grammar.add_rule('START', 'presup_', ['False', 'BOOL'], 1.0)

grammar.add_rule('START', 'presup_', ['False', 'False'], 1.0)
grammar.add_rule('START', 'presup_', ['True', 'True'], 1.0)

grammar.add_rule('BOOL', 'not_', ['BOOL'], 1.0)
grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
grammar.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
#grammar.add_rule('BOOL', 'nonempty_', ['SET'], 1.0) # don't need this if we do logical operations

grammar.add_rule('BOOL', 'empty_', ['SET'], 1.0)
grammar.add_rule('BOOL', 'subset_', ['SET', 'SET'], 1.0)
grammar.add_rule('BOOL', 'exhaustive_', ['SET', 'context.S'], 1.0)
grammar.add_rule('BOOL', 'cardinality1_', ['SET'], 1.0) # if cardinalities are included, don't include these!
grammar.add_rule('BOOL', 'cardinality2_', ['SET'], 1.0)
grammar.add_rule('BOOL', 'cardinality3_', ['SET'], 1.0)

grammar.add_rule('SET', 'union_', ['SET', 'SET'], 1.0)
grammar.add_rule('SET', 'intersection_', ['SET', 'SET'], 1.0)
grammar.add_rule('SET', 'setdifference_', ['SET', 'SET'], 1.0)

# These will just be attributes of the current context
grammar.add_rule('SET', 'context.A', None, 10.0)
grammar.add_rule('SET', 'context.B', None, 10.0)
grammar.add_rule('SET', 'context.S', None, 10.0) ## Must include this or else we can't get complement

# Cardinality operations
grammar.add_rule('BOOL', 'cardinalityeq_', ['SET', 'SET'], 1.0)
grammar.add_rule('BOOL', 'cardinalitygt_', ['SET', 'SET'], 1.0)
#grammar.add_rule('BOOL', 'cardinalityeq_', ['SET', 'CARD'], 1.0)
#grammar.add_rule('BOOL', 'cardinalitygt_', ['SET', 'CARD'], 1.0)
#grammar.add_rule('BOOL', 'cardinalitygt_', ['CARD', 'SET'], 1.0)
###grammar.add_rule('CARD', 'cardinality_', ['SET'], 1.0)

##grammar.add_rule('CARD',  '0', None, 1.0)
#grammar.add_rule('CARD',  '1', None, 1.0)
#grammar.add_rule('CARD',  '2', None, 1.0)
#grammar.add_rule('CARD',  '3', None, 1.0)
#grammar.add_rule('CARD',  '4', None, 1.0)
#grammar.add_rule('CARD',  '5', None, 1.0)
#grammar.add_rule('CARD',  '6', None, 1.0)
#grammar.add_rule('CARD',  '7', None, 1.0)
#grammar.add_rule('CARD',  '8', None, 1.0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Define a test set -- for doign Gricean things
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TESTING_SET_SIZE = 1000

# create a small list of all plausible context sets.
# NOTE: Do no use this if utterances consist of all possible words (e.g. man/pirate are allowed to vary)
all_possible_context_sets = []
APD_N = 6
for adb in xrange(APD_N):
    for bda in xrange(APD_N-adb):
        for anb in xrange(APD_N-adb-bda):
            for s in xrange(APD_N-adb-bda-anb):
                adb_ = set([Obj(shape="man", job="chef") for i in xrange(adb)])
                bda_ = set([Obj(shape="woman", job="pirate") for i in xrange(bda)])
                anb_ = set([Obj(shape="man", job="pirate") for i in xrange(anb)])
                s_   = set([Obj(shape="woman", job="chef") for i in xrange(s)])

                all_possible_context_sets.append( [adb_.union(anb_), bda_.union(anb_), s_])


TESTING_SET = [MyContext(A=x[0], B=x[1], S=x[2]) for x in all_possible_context_sets] # [ sample_context_set() for x in xrange(TESTING_SET_SIZE)  ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Functions for gricean
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_my_hypothesis():
    return LOTHypothesis(grammar, args=['context'])

from cachetools import lru_cache

@lru_cache
def my_weight_function(h):
    return gricean_weight(h, TESTING_SET)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Define the target
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Primitives.Semantics import *
## Write this out as a dictionary so that we can load it into a GriceanSimpleLexicon easier
target_functions = { 'every' : lambda context: presup_(nonempty_(context.A), subset_(context.A,context.B)),\
                     'some'  : lambda context: presup_(nonempty_(context.A), nonempty_(intersection_(context.A,context.B))),\
                     'a': lambda context: presup_(True, nonempty_(intersection_(context.A,context.B))),\
                     'the': lambda context: presup_(cardinality1_(context.A), subset_(context.A,context.B)),\
                     'no': lambda context: presup_(nonempty_(context.A), empty_(intersection_(context.A,context.B))),\
                     'both': lambda context: presup_(cardinality2_(context.A), subset_(context.A,context.B)),\
                     'neither':lambda context: presup_(cardinality2_(context.A), empty_(intersection_(context.A,context.B))),\
                     #'either': lambda context: presup_(cardinality2_(context.A), cardinality1_(intersection_(context.A,context.B))),\
                     #'one':lambda context: presup_(True, cardinality1_(intersection_(context.A,context.B))),\
                     #'two':lambda context: presup_(True, cardinality2_(intersection_(context.A,context.B))),\
                     #'three':lambda context: presup_(True, cardinality3_(intersection_(context.A,context.B))),\
                     #'most':lambda context: presup_(nonempty_(context.A), cardinalitygt_(intersection_(context.A,context.B), setdifference_(context.A,context.B)))

                     #'few':lambda context: presup_(True, cardinalitygt_(3, intersection_(context.A,context.B))),
                     #'many':lambda context: presup_(True, cardinalitygt_(intersection_(context.A,context.B), 3)),
                     #'half':lambda context: presup_(nonempty_(context.A), cardinalityeq_(intersection_(context.A,context.B), setdifference_(context.A,context.B)))
                     }

target = GriceanQuantifierLexicon(make_my_hypothesis, my_weight_function)

for w, f in target_functions.items():
    target.set_word(w, LOTHypothesis(grammar, value='SET_IN_TARGET', f=f))
