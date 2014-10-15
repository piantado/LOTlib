from LOTlib.Miscellaneous import random, weighted_sample
from LOTlib.DataAndObjects import FunctionData, sample_sets_of_objects, make_all_objects
from LOTlib.Evaluation.Primitives.Number import word_to_number
from Global import ALPHA, WORDS

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
    resp = [ ne(set(sample_sets_of_objects(n, all_objects))) for n in xrange(1, 10)]
    return ''.join([str(word_to_number[x]) if (x is not None and x is not 'undef') else 'U' for x in resp])


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
        data.append(FunctionData(input=[s], output=r))  # convert to "FunctionData" and store
    return data


# # # # # # # # # # # # # # # # # # # # # # # # #
# All objects -- not very exciting

#here this is really just a dummy -- one type of object, which is replicated in sample_sets_of_objects
all_objects = make_all_objects(shape=['duck'])

# all possible data sets on 10 objects
all_possible_data = [ ('', set(sample_sets_of_objects(n, all_objects))) for n in xrange(1,10) ]
