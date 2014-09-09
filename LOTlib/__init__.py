
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
SIG_INTERRUPTED = False
def signal_handler(signal, frame):
    global SIG_INTERRUPTED
    SIG_INTERRUPTED = True

# Handle interrupt CTRL-C
signal.signal(signal.SIGINT, signal_handler)

# handle signal 24, CPU time exceeded (via slurm or other cluster managers)
signal.signal(24, signal_handler)



def lot_iter(g, multi_break=False):
    """
            A wrapper that lets you wrap a generater, rather than have to write "if LOTlib.SIG_INTERRUPTED..."
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
