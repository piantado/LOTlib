"""

	Parse simplified lambda expressions--enough for ATIS. 
	
	TODO: No quoting etc. supported yet. 
	
	NOTE: This is based on http://pyparsing.wikispaces.com/file/view/sexpParser.py/278417548/sexpParser.py
	
"""


from pyparsing import *
from LOTlib.FunctionNode import *
#from LOTlib.FunctionNode import list2FunctionNode
import pprint

#####################################################################
## Here we define a super simple grammar for lambdas

LPAR, RPAR, LBRK, RBRK = map(Suppress, "()[]")
token = Word(alphanums + "-./_:*+=!<>$")

sexp = Forward()
sexpList = Group(LPAR + ZeroOrMore(sexp) + RPAR)
sexp << ( token | sexpList )

#####################################################################

def simpleLambda2List(s):
	"""
		Return a list of list of lists... for a string containing a simple lambda expression (no quoting, etc)
		NOTE: This converts to lowercase
	"""
	
	x = sexp.parseString(s, parseAll=True)
	x = x[0] # remove that first element
	x = x.asList() # get back as a list rather than a pyparsing.ParseResults
	return x

def simpleLambda2FunctionNode(s, style="atis"):
	"""
		Take a string for lambda expressions and map them to a real FunctionNode tree
	"""
	return list2FunctionNode(simpleLambda2List(s), style=style)
	
if __name__ == '__main__':

	test1 = "(defun factorial (x) (if (= x 0) 1 (* x (factorial (- x 1)))))"
	test2 = "(lambda $0 e (and (day_arrival $0 thursday:da) (to $0 baltimore:ci) (< (arrival_time $0) 900:ti) (during_day $0 morning:pd) (exists $1 (and (airport $1) (from $0 $1)))))"
	
	y = simpleLambda2List(test1)
	#print y
	assert str(y) == str(['defun', 'factorial', ['x'], ['if', ['=', 'x', '0'], '1', ['*', 'x', ['factorial', ['-', 'x', '1']]]]])
	
	x = simpleLambda2List(test2)
	#print x
	assert str(x) == str(['lambda', '$0', 'e', ['and', ['day_arrival', '$0', 'thursday:da'], ['to', '$0', 'baltimore:ci'], ['<', ['arrival_time', '$0'], '900:ti'], ['during_day', '$0', 'morning:pd'], ['exists', '$1', ['and', ['airport', '$1'], ['from', '$0', '$1']]]]])
	
	print list2FunctionNode(x)
	