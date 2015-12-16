import sys

def load_example(model):
    """
    Loads a model, returning make_hypothesis, make_data, make_sampler
    """
    exec("from LOTlib.Examples.%s import make_hypothesis, make_data" % model)

    return make_hypothesis, make_data