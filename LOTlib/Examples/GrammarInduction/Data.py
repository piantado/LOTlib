

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The data here has a form of {string:count}. The likelihood is computed by generating forward and estimating
# the likelihood of the data, running a generator forward

from LOTlib.DataAndObjects import FunctionData

# # The data here has a form
def make_data(size=50):
    return [FunctionData(input=[],
                         output={'N V': size,
                                 'D N V': size,
                                 'D N V N': size,
                                 'D N V D N': size})]
