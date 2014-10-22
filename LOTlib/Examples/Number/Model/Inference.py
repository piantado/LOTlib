# -*- coding: utf-8 -*-
"""
        Shared functions and variables for the number learning model.
"""

from LOTlib.Miscellaneous import log1mexp
from Grammar import grammar
from Hypothesis import NumberExpression

ALPHA = 0.75 # the probability of uttering something true
GAMMA = -30.0 # the log probability penalty for recursion
LG_1MGAMMA = log1mexp(GAMMA)
MAX_NODES = 50 # How many FunctionNodes are allowed in a hypothesis? If we make this, say, 20, things may slow down a lot

WORDS = ['one_', 'two_', 'three_', 'four_', 'five_', 'six_', 'seven_', 'eight_', 'nine_', 'ten_']

# # # # # # # # # # # # # # # # # # # # # # # # #
# Standard exports

def make_h0(**kwargs):
    return NumberExpression(grammar, **kwargs)


# if __name__ == "__main__":
#
#     for _ in xrange(1000):
#         h = NumberExpression()
#         print get_knower_pattern(h), h