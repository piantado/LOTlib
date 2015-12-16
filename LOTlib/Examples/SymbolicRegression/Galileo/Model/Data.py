from LOTlib.DataAndObjects import FunctionData

"""
This uses Galileo's data on a falling ball.

See: http://www.amstat.org/publications/jse/v3n1/datasets.dickey.html
See also: Jeffreys, W. H., and Berger, J. O. (1992), "Ockham's Razor and Bayesian Analysis," American
    Scientist, 80, 64-72 (Erratum, p. 116).
"""

# NOTE: these must be floats, else we get hung up on powers of ints
data_sd = 50.0

def make_data(n=1):
    return [ FunctionData(input=[1000.], output=1500., ll_sd=data_sd),
             FunctionData(input=[828.], output=1340., ll_sd=data_sd),
             FunctionData(input=[800.], output=1328., ll_sd=data_sd),
             FunctionData(input=[600.], output=1172., ll_sd=data_sd),
             FunctionData(input=[300.], output=800., ll_sd=data_sd),
             FunctionData(input=[0.], output=0., ll_sd=data_sd) # added 0,0 since it makes physical sense.
    ]*n