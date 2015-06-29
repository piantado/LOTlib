"""

The data here has a form of {string:count}. The likelihood is computed by generating forward and estimating
the likelihood of the data, running a generator forward

"""

NDATA = 50  # How many of each data point have we seen?

from LOTlib.DataAndObjects import FunctionData

# # The data here has a form
def make_data():
    return [FunctionData(input=[],
                         output={'N V': NDATA,
                                 'D N V': NDATA,
                                 'D N V N': NDATA,
                                 'D N V D N': NDATA})]