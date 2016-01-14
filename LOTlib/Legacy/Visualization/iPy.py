"""
Tools for visualizing useful things.

>>> import LOTlib.Visualization as viz

"""
from IPython.display import clear_output
import sys

def print_iters(i, num_iters, increm=20):
    """Print incremental statements as we generate a large number of hypotheses.

    TODO: should this be made into a more general version?

    """
    i += 1
    j = 0
    if i % (num_iters/increm) == 0:
        j += 1
        clear_output()
        print '\nGenerating %i hypotheses...\n' % i
        print '[' + '#'*j + '-'*(increm-j) + ']'
        sys.stdout.flush()

