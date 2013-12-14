"""
	Read atis, parse its lambda expressions, and print a co-occurance between subtrees and words
	
	TODO: Need to iterate through the leaves too, to get tihngs like" minneapolis"
"""

from LOTlib.Miscellaneous import *
from LOTlib.SimpleLambdaParser import *
from LOTlib.Grammar import Grammar
import re

ATIS_path = "/home/piantado/Desktop/mit/Corpora/atis/atis.dev"

G = Grammar()
co_occurance_matrix = dict()

txt = None
for l in open(ATIS_path, 'r'):
	l = l.strip()
	
	if re.match("^\s*\(", l):  # a lambda expression
		
		lam = simpleLambda2FunctionNode(l, style="atis")
		
		#for uc in G.all_simple_uncompositions(lam):
			#print uc
		#print "\n\n"
		
		for w in re.split("\s",txt):
			for t in lam:
				d = co_occurance_matrix.get(w,dict())
				d[t] = d.get(t,0) + 1 
				co_occurance_matrix[w] = d # update
				
	elif re.search("[a-zA-Z]",l):  # a sentence
		txt = l # text comes before so just store	
		
for w in co_occurance_matrix.keys():
	for t in sorted( co_occurance_matrix[w].keys(), key=lambda x:co_occurance_matrix[w][x]):
		print w, "\t", co_occurance_matrix[w][t], "\t", t
		
