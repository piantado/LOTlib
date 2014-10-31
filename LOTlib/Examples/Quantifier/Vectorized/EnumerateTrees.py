# -*- coding: utf-8 -*-

"""
Enumerate all possible trees (as FunctionHypotheses), saving them to a file (for later gibbs as in Search_UnoptimizedGibbs and Search_VectorizedGibbs)

"""
import sys

from Utilities import *

OUT = "data/all_trees_2012May2.pkl"
DEPTH = 3
MAX_NODES = 11

all_tree_count = 0

## Collapse trees by how they act on data -- collapse equivalent functions
collapsed_forms = dict()

# A function to collapse trees together based on their functional response
def add_to_collapsed_trees(t):
    resps = ';'.join(map(str, get_tree_set_responses(t, all_possible_context_sets)))

    tprior = t.log_probability()

    if resps in collapsed_forms: # add to the existing collapsed form if no recursion
        collapsed_forms[resps].my_log_probability = logplusexp( collapsed_forms[resps].log_probability(), tprior )
        if tprior > collapsed_forms[resps].display_tree_probability: # display the most concise form
            collapsed_forms[resps] = t
            collapsed_forms[resps].display_tree_probability = tprior
    else:
        collapsed_forms[resps] = t
        collapsed_forms[resps].display_tree_probability = tprior
        t.my_log_probability = tprior # FunctionNode uses this value when we call log_probability()
        print ">>", all_tree_count, len(collapsed_forms),  t, tprior

############################################
### Now actually enumarate trees
for t in grammar.increment_tree('START',DEPTH):
    if t.count_subnodes() <= MAX_NODES:
        add_to_collapsed_trees(t)
        all_tree_count += 1
        print ">", t, t.posterior_score, t.log_probability()

## for kinder saving and unsaving:
upq = FiniteBestSet(max=True)
for k in collapsed_forms.values():
    upq.push(LOTHypothesis(grammar, k, args=['A', 'B', 'S']), 0.0)
upq.save(OUT)

print "Total tree count: ", all_tree_count
