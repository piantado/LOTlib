# -*- coding: utf-8 -*-

from Data import get_knower_pattern
from Grammar import grammar
from Hypothesis import NumberExpression


def make_h0(**kwargs):
    return NumberExpression(grammar, **kwargs)


if __name__ == "__main__":
    for _ in xrange(1000):
        h = NumberExpression()
        print get_knower_pattern(h), h