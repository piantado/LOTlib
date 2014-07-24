# Git Bisect test script
# Given a Python script, 

# modified from http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
import signal

def handler(signum, frame):
	return 0

def function_to_test():
	from LOTlib.Examples.NAND import DoInference

signal.signal(signal.SIGALRM, handler)
signal.alarm(5) # abort the function to test after 5 seconds

function_to_test()
