"""
	A simple case of combinatory categorial grammar for a toy domain.
	
	This just uses brute force parsing. 
	
	
	TODO: Learn that MAN is JOHN or BILL
	
"""
from copy import copy

import LOTlib
from LOTlib import lot_iter
from LOTlib.Miscellaneous import unique, qq
from LOTlib.Grammar import Grammar
from LOTlib.DataAndObjects import UtteranceData
from LOTlib.FunctionNode import cleanFunctionNodeString
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib.Hypotheses.SimpleLexicon import SimpleLexicon
from LOTlib.Inference.MetropolisHastings import mh_sample
from LOTlib.FiniteBestSet import FiniteBestSet
from Shared import *
from CCGLexicon import CCGLexicon


SEMANTIC_1PREDICATES = ['SMILED', 'LAUGHED', 'MAN', 'WOMAN']
SEMANTIC_2PREDICATES = ['SAW', 'LOVED']
OBJECTS              = ['JOHN', 'MARY', 'SUSAN', 'BILL']

G = Grammar()

G.add_rule('START', '', ['FUNCTION'], 2.0) 
G.add_rule('START', '', ['BOOL'], 1.0) 
G.add_rule('START', '', ['OBJECT'], 1.0) 

for m in SEMANTIC_1PREDICATES: 
	G.add_rule('BOOL', 'C.relation_', [ qq(m), 'OBJECT'], 1.0)
	
for m in SEMANTIC_2PREDICATES: 
	G.add_rule('BOOL', 'C.relation_', [ qq(m), 'OBJECT', 'OBJECT'], 1.0)

for o in OBJECTS:
	G.add_rule('OBJECT', qq(o), None, 1.0)

G.add_rule('BOOL', 'exists_', ['FUNCTION.O2B', 'C.objects'], 1.00) # can quantify over objects->bool functions
G.add_rule('BOOL', 'forall_', ['FUNCTION.O2B', 'C.objects'], 1.00)
G.add_rule('FUNCTION.O2B', 'lambda', ['BOOL'], 1.0, bv_type='OBJECT') 

G.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
G.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)
G.add_rule('BOOL', 'not_', ['BOOL'], 1.0)

# And for outermost functions
G.add_rule('FUNCTION', 'lambda', ['START'], 1.0, bv_type='OBJECT')
G.add_rule('FUNCTION', 'lambda', ['START'], 1.0, bv_type='BOOL', bv_args=['OBJECT'])
G.add_rule('FUNCTION', 'lambda', ['START'], 1.0, bv_type='BOOL', bv_args=['OBJECT', 'OBJECT'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up the data

possible_utterances = [] # this will be referenced in every UTteranceData, and at the end we'll use it to do all possible strings

data = [] # For now, some unambiguous data:
data.append(  UtteranceData( utterance=str2sen('john saw mary'), context=Context(OBJECTS, [("SAW", "JOHN", "MARY")]), possible_utterances=possible_utterances  ))
data.append(  UtteranceData( utterance=str2sen('mary saw john'), context=Context(OBJECTS, [("SAW", "MARY", "JOHN")]), possible_utterances=possible_utterances  ))

data.append(  UtteranceData( utterance=str2sen('mary smiled'), context=Context(OBJECTS, [("SMILED", "MARY")]), possible_utterances=possible_utterances  ))
data.append(  UtteranceData( utterance=str2sen('john smiled'), context=Context(OBJECTS, [("SMILED", "JOHN")]), possible_utterances=possible_utterances  ))
data.append(  UtteranceData( utterance=str2sen('bill smiled'), context=Context(OBJECTS, [("SMILED", "BILL")]), possible_utterances=possible_utterances  ))
data.append(  UtteranceData( utterance=str2sen('susan smiled'), context=Context(OBJECTS, [("SMILED", "SUSAN")]), possible_utterances=possible_utterances  ))

data.append(  UtteranceData( utterance=str2sen('john is man'), context=Context(OBJECTS, [("MAN", "JOHN")]), possible_utterances=possible_utterances  ))
data.append(  UtteranceData( utterance=str2sen('bill is man'), context=Context(OBJECTS, [("MAN", "BILL")]), possible_utterances=possible_utterances  ))
data.append(  UtteranceData( utterance=str2sen('mary is woman'), context=Context(OBJECTS, [("WOMAN", "MARY")]), possible_utterances=possible_utterances  ))
data.append(  UtteranceData( utterance=str2sen('susan is woman'), context=Context(OBJECTS, [("WOMAN", "SUSAN")]), possible_utterances=possible_utterances  ))


#data.append(  UtteranceData( utterance=str2sen('every man smiled'), context=Context(OBJECTS, [("SMILED", "JOHN"),("SMILED", "BILL")]), possible_utterances=possible_utterances  ))
#data.append(  UtteranceData( utterance=str2sen('every woman smiled'), context=Context(OBJECTS, [("SMILED", "MARY"),("SMILED", "SUSAN")]), possible_utterances=possible_utterances  ))
#data.append(  UtteranceData( utterance=str2sen('every man laughed'), context=Context(OBJECTS, [("LAUGHED", "JOHN"),("LAUGHED", "BILL")]), possible_utterances=possible_utterances  ))
#data.append(  UtteranceData( utterance=str2sen('every woman laughed'), context=Context(OBJECTS, [("LAUGHED", "MARY"),("LAUGHED", "SUSAN")]), possible_utterances=possible_utterances  ))


# Just treat each possible utterance as 
for di in data: possible_utterances.append( di.utterance )


# keep track of all the words
all_words = set()
for di in data: 
	for w in di.utterance: all_words.add(w)

possible_utterances.append( str2sen('mary smiled'))
all_words.add('mary')

# How we make a hypothesis inside the lexicon
def make_hypothesis(): 
	return LOTHypothesis(G, args=['C'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SAMPLES = 100000

def run(llt=1.0):
	
	h0 = CCGLexicon(make_hypothesis, words=all_words, alpha=0.9, palpha=0.9, likelihood_temperature=llt)

	fbs = FiniteBestSet(N=10)
	for h in lot_iter(mh_sample(h0, data, SAMPLES)):
		fbs.add(h, h.posterior_score)
	
	return fbs


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### MPI map
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from SimpleMPI.MPI_map import MPI_map, is_master_process

allret = MPI_map(run, map(lambda x: [x], [0.01, 0.1, 1.0] * 100 )) 

if is_master_process():

	allfbs = FiniteBestSet(max=True)
	allfbs.merge(allret)

	H = allfbs.get_all()
	
	for h in H:
		h.likelihood_temperature = 0.1 # on what set of data we want?
		h.compute_posterior(data)

	# show the *average* ll for each hypothesis
	for h in sorted(H, key=lambda h: h.posterior_score):
		print h.posterior_score, h.prior, h.likelihood, h.likelihood_temperature
		print h


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Run on a single computer, printing out
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#fbs = FiniteBestSet(N=100)
#h0 = CCGLexicon(make_hypothesis, words=all_words, alpha=0.9, palpha=0.9, likelihood_temperature=0.051)
#for i, h in lot_iter(enumerate(mh_sample(h0, data, 400000000, skip=0, debug=False))):
	#fbs.add(h, h.posterior_score)
	
	#if i%100==0:
		#print h.posterior_score, h.prior, h.likelihood #, re.sub(r"\n", ";", str(h))
		#print h

#for h in fbs.get_all(sorted=True):
	#print h.posterior_score, h.prior, h.likelihood #, re.sub(r"\n", ";", str(h))
	#print h

	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Just generate and parse
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#for _ in xrange(1000):

	#cp = h.can_parse(data[3].utterance)
	#if cp:
		#s, t, f = cp
		#print L
		#print s, t, f
		#print f(data[3].context)
		#print "\n\n"
