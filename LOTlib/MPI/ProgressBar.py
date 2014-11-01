# -*- coding: utf-8 -*-
import random
import sys

last_progress_flipper = False
def draw_progress(pct, ncols=20, ret="\r"):

    global last_progress_flipper

    out = ret+"# ["

    for i in xrange(ncols):

        if float(i)/float(ncols) < pct: out = out+"-"
        else:                           out = out+" "
    out = out+"]"

    if last_progress_flipper: out = out+" ;-) "
    else:                     out = out+" :-) "

    last_progress_flipper = not last_progress_flipper

    out = out+(" %0.1f"%(100.0*pct)) + "%"

    print >>sys.stderr, out,
    sys.stderr.flush()
