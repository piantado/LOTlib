# -*- coding: utf-8 -*- 
from subprocess import Popen, PIPE
import nltk.tree

"""
A module for generating graphviz images of lisp-y tree strings

taken from :  https://github.com/pln-fing-udelar/pln_inco
"""

def generate(entrada,format):
	""" Given a graphviz specification, returns the dot generated file, in the specifyied format
	Format can be one of: 'jpg', 'png'
	"""
	
	if format=='jpg':
		opt='-Tjpg'
	elif format=='png':
		opt='-Tpng'
		
	p=Popen(['dot',opt], stdin=PIPE, stdout=PIPE)
	return p.communicate(input=entrada)[0]


def tree_to_dot(t):
	"""
	Given a NLTK full syntax Tree, returns a dot representation, suitable for using with Graphviz.
	This function assumes that the node property is a String

	type t: L{nltk.tree.Tree}
	rtype: C{string}

	"""


	def gv_print(t,start_node=1):
		"""
		Print the tree for a defined node. Nodes are specified in-order in the original tree	
		"""

		
		# Print the start node of the tree
		s ='%s [label="%s"]' % (start_node,t.node)
		pos=start_node+1

		# Print the node's children
		for child in t:
			if isinstance(child,nltk.tree.Tree):
				(s_child,newpos)=gv_print(child,pos)
				s=s+'\n'+ s_child
				s=s+'\n%s -> %s' % (start_node,pos)
				pos=newpos
			elif isinstance(child, str):
				s=s+'\n%s [label="%s", shape=plaintext]' % (pos,child)
				s=s+'\n%s -> %s' % (start_node,pos)	
			pos+=1
		return (s,pos-1)

	# Print the digraph dot specification
	s='digraph G{\n'	
	s+='edge [dir=none]\n'
	s+='node [shape=plaintext]\n'
	
	s+=gv_print(t)[0]
	s+="\n}"

	return s

