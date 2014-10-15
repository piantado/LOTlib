"""
Module for working with  NLTK syntax trees...

taken from :  https://github.com/pln-fing-udelar/pln_inco
"""
import nltk.tree


def tree_to_dot(t):
	"""
	Given a NLTK full syntax Tree, returns a dot representation, suitable for using with Graphviz.
	This function assumes that the node property is a String

	@type t: L{nltk.tree.Tree}
	@rtype: C{string}

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


def dependency_to_dot(t):
	"""
	Given a NLTK representation for a dependency analysis (such as the one in the CoNLL 2007 corpus, returns a dot representation
	suitable for using with Graphviz
	@type t:L{nltk.parse.dependencygraph.DependencyGraph}
	@rtype C{String}
	"""

	# Start the digraph specification	
	s='digraph G{\n'
	s+='edge [dir=forward]\n'
	s+='node [shape=plaintext]\n'

	# Draw the remaining nodes
	for node in t.nodelist[1:]:
		s+='\n%s [label="%s (%s)"]' %  (node['address'],node['address'],node['word'])
		if node['head']:
			if node['rel'] != '_':
				s+= '\n%s -> %s [label="%s"]' % (node['head'],node['address'],node['rel'])
			else:
				s+= '\n%s -> %s ' % (node['head'],node['address'])

        s+="\n}"

	return s

def stanford_dependency_to_malt_tab(stanford_basic_dep,words_and_pos):
	"""
	Given a list of dependendencies, output from the Stanford parser, using basicDependencies, and a list of (word,POS) pairs
	Returns a Malt-TAB compliant specification, suitable to be read using nltk.dependencygraph.DependencyGraph
	@rtype C{String}
	@type stanford_basic_dep:C{string}
	@type words_and_pos:C{List}
	"""
	import re
	malt_tab_rep=''
	words=[word for (word,pos) in words_and_pos]
	pos=[pos for (word,pos) in words_and_pos]

	rels=stanford_basic_dep.split('\n')
    	xs=[re.match(r'(\w+)\((\w+)-(\d+), (\w+)-(\w+).*',r).groups() for r in rels] 
    	ys= sorted(xs,key=lambda elem:int(elem[4]))
    	i=0
    	for y in ys:
        	i+=1
        	while i<int(y[4]):
            		# Fix a missing word
		        malt_tab_rep+="{}\t{}\t{}\t{}\n".format(words[i-1],pos[i-1],None,None)    
		        i+=1
        	malt_tab_rep+="{}\t{}\t{}\t{}\n".format(words[i-1],pos[i-1],y[2],y[0])
	return malt_tab_rep
