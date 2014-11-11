
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Store a version number
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LOTlib_VERSION = "0.2.0"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This allows us to use the variable SIG_INTERRUPTED inside loops etc
# to catch breaks.
# import LOTlib
# if LOTlib.SIG_INTERRUPTED: ...
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import signal
import sys
from Inference.MetropolisHastings import MHSampler, mh_sample
from DataAndObjects import FunctionData, UtteranceData, make_all_objects
from Evaluation import Eval

SIG_INTERRUPTED = False
def signal_handler(signal, frame):
    global SIG_INTERRUPTED
    SIG_INTERRUPTED = True
    print >>sys.stderr, "# Signal %s caught."%signal

# Handle interrupt CTRL-C
signal.signal(signal.SIGINT, signal_handler)


def lot_iter(g, multi_break=False):
    """Easy way to ctrl-C out of a loop.

    Lets you wrap a generater, rather than have to write "if LOTlib.SIG_INTERRUPTED..."

    """

    import LOTlib # WOW, this is weird scoping, but it doesn't work if you treat this as a local variable (you can't from LOTlib import lot_iter)

    for x in g:
        #global SIG_INTERRUPTED
        if LOTlib.SIG_INTERRUPTED:

            # reset if we should
            if not multi_break: SIG_INTERRUPTED = False

            # and break
            break
        else:

            yield x
