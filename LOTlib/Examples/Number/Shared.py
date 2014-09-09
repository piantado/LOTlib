# -*- coding: utf-8 -*-
"""
        Shared functions and variables for the number learning model.
"""

from LOTlib.Grammar import Grammar
from LOTlib.Miscellaneous import *
from LOTlib.DataAndObjects import FunctionData, sample_sets_of_objects, make_all_objects
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Evaluation.Primitives.Number import word_to_number

ALPHA = 0.75 # the probability of uttering something true
GAMMA = -30.0 # the log probability penalty for recursion
LG_1MGAMMA = log1mexp(GAMMA)
MAX_NODES = 50 # How many FunctionNodes are allowed in a hypothesis? If we make this, say, 20, things may slow down a lot

WORDS = ['one_', 'two_', 'three_', 'four_', 'five_', 'six_', 'seven_', 'eight_', 'nine_', 'ten_']

########################################################################################################
## Define a PCFG

## The priors here are somewhat hierarchical by type in generation, tuned to be a little more efficient
## (but the actual RR prior does not care about these probabilities)

grammar = Grammar(start='WORD')

grammar.add_rule('BOOL', 'and_',    ['BOOL', 'BOOL'], 1./3.)
grammar.add_rule('BOOL', 'or_',     ['BOOL', 'BOOL'], 1./3.)
grammar.add_rule('BOOL', 'not_',    ['BOOL'], 1./3.)

grammar.add_rule('BOOL', 'True',    None, 1.0/2.)
grammar.add_rule('BOOL', 'False',   None, 1.0/2.)

## note that this can take basically any types for return values
grammar.add_rule('WORD', 'if_',    ['BOOL', 'WORD', 'WORD'], 0.5)
grammar.add_rule('WORD', 'ifU_',    ['BOOL', 'WORD'], 0.5) # if returning undef if condition not met

grammar.add_rule('BOOL', 'cardinality1_',    ['SET'], 1.0)
grammar.add_rule('BOOL', 'cardinality2_',    ['SET'], 1.0)
grammar.add_rule('BOOL', 'cardinality3_',    ['SET'], 1.0)

grammar.add_rule('BOOL', 'equal_',    ['WORD', 'WORD'], 1.0)

grammar.add_rule('SET', 'union_',     ['SET', 'SET'], 1./3.)
grammar.add_rule('SET', 'intersection_',     ['SET', 'SET'], 1./3.)
grammar.add_rule('SET', 'setdifference_',     ['SET', 'SET'], 1./3.)
grammar.add_rule('SET', 'select_',     ['SET'], 1.0)

grammar.add_rule('SET', 'x',     None, 4.0)

grammar.add_rule('WORD', 'L_',        ['SET'], 1.0)

grammar.add_rule('WORD', 'next_', ['WORD'], 1.0)
grammar.add_rule('WORD', 'prev_', ['WORD'], 1.0)

#grammar.add_rule('WORD', 'undef', None, 1.0)
# These are quoted (q) since they are strings when evaled
grammar.add_rule('WORD', q('one_'), None, 0.10)
grammar.add_rule('WORD', q('two_'), None, 0.10)
grammar.add_rule('WORD', q('three_'), None, 0.10)
grammar.add_rule('WORD', q('four_'), None, 0.10)
grammar.add_rule('WORD', q('five_'), None, 0.10)
grammar.add_rule('WORD', q('six_'), None, 0.10)
grammar.add_rule('WORD', q('seven_'), None, 0.10)
grammar.add_rule('WORD', q('eight_'), None, 0.10)
grammar.add_rule('WORD', q('nine_'), None, 0.10)
grammar.add_rule('WORD', q('ten_'), None, 0.10)

##########################################################
#Define a class for running MH

class NumberExpression(LOTHypothesis):
    
    def __init__(self, grammar, value=None, f=None, proposal_function=None, **kwargs):
        LOTHypothesis.__init__(self, grammar, value=value, proposal_function=proposal_function, **kwargs)

    def compute_prior(self):
        """
                Compute the number model prior: log_probability() with a penalty on whether or not recursion is used
        """
        recursion_penalty = 0
        if self.value.count_nodes() > MAX_NODES:
            self.prior = -Infinity
        else:
            if self.value.contains_function("L_"): 
                recursion_penalty = GAMMA
            else:
                recursion_penalty = LG_1MGAMMA

        self.prior = (recursion_penalty + self.value.log_probability())  / self.prior_temperature
        self.posterior_score = self.prior + self.likelihood

        return self.prior

    def compute_single_likelihood(self, datum):
        """
                Computes the likelihood of data.
                TODO: Make sure this precisely matches the number paper.

        """
        response = self(*datum.input)
        if response == 'undef' or response == None:
            return log(1.0/10.0) # if undefined, just sample from a base distribution
        else:
            return log( (1.0 - ALPHA)/10.0 + ALPHA * ( response == datum.output ) )



# # # # # # # # # # # # # # # # # # # # # # # # #
# The target

#target = NumberExpression("one_ if cardinality1_(x) else next_(L_(setdifference_(x, select_(x))))") # we need to translate "if" ourselves
#target = NumberExpression(value="if_(cardinality1_(x), one_, two_)")
# NOTE: Not necessary, but only for testing -- these are discovered in the real model via search
#one_knower   = NumberExpression("one_ if cardinality1_(x) else undef")
#two_knower   = NumberExpression("one_ if cardinality1_(x) else ( two_ if cardinality2_(x) else undef )")
#three_knower = NumberExpression("one_ if cardinality1_(x) else ( two_ if (cardinality2_(x) ) else ( three_ if (cardinality3_(x) else undef) )")
#four_knower  = NumberExpression("one_ if cardinality1_(x) else ( two_ if (cardinality2_(x) ) else ( three_ if (cardinality3_(x) else (four_ if (cardinality4_(x) else undef) ) )")

def get_knower_pattern(ne):
    """
            This computes a string describing the behavior of this knower-level
    """
    out = ''
    resp = [ ne(set(sample_sets_of_objects(n, all_objects))) for n in xrange(1,10)]
    return ''.join([ str(word_to_number[x]) if (x is not None and x is not 'undef' ) else 'U' for x in resp])


def generate_data(data_size):
    """
            Sample some data according to the target
    """
    data = []
    for i in range(data_size):
        # how many in this set
        set_size = weighted_sample( range(1,10+1), probs=[7187, 1484, 593, 334, 297, 165, 151, 86, 105, 112] )
        # get the objects in the current set
        s = set(sample_sets_of_objects(set_size, all_objects))

        # sample according to the target
        if random() < ALPHA: r = WORDS[len(s)-1]
        else:                r = weighted_sample( WORDS )

        # and append the sampled utterance
        data.append(FunctionData( input=[s], output=r) ) # convert to "FunctionData" and store
    return data


# # # # # # # # # # # # # # # # # # # # # # # # #
# All objects -- not very exciting

#here this is really just a dummy -- one type of object, which is replicated in sample_sets_of_objects
all_objects = make_all_objects(shape=['duck'])

# all possible data sets on 10 objects
all_possible_data = [ ('', set(sample_sets_of_objects(n, all_objects))) for n in xrange(1,10) ]

# # # # # # # # # # # # # # # # # # # # # # # # #
# Standard exports

def make_h0(**kwargs):
    return NumberExpression(grammar, **kwargs)


if __name__ == "__main__":
    
    for _ in xrange(1000):
        h = NumberExpression()
        print get_knower_pattern(h), h