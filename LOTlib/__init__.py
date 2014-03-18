
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Store a version number
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LOTlib_VERSION = "0.1.0"

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
signal.signal(signal.SIGINT, signal_handler)


def lot_iter(g, multi_break=False):
	"""
		A wrapper that lets you wrap a generater, rather than have to write "if LOTlib.SIG_INTERRUPTED..."
	"""
	for x in g:
		
		if LOTlib.SIG_INTERRUPTED: 
			
			# reset if we should
			if not multi_break: LOTlib.SIG_INTERRUPTED = False 
			
			# and break
			break
		else:
			 
			yield g
		