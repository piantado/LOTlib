"""
    The intuition here is that one way to prevent chains from duplicating work is to divide up the space and ensure chains stay 
    in separate regions. This is accomplished here by setting the resample_p to very tiny except for the leaves, after
    enumerating up to some depth in the grammar. Warning: the number of trees (and thus chains) will likely be exponential in the depth!
    
    This is at present NOT a correct sampler. I think it could easily be made one by keeping track of how often a chain in one
    partition proposes a change to each other partition. 

"""
from LOTlib.Miscellaneous import Infinity
from copy import copy


from MultipleChainMCMC import MultipleChainMCMC

from LOTlib.Subtrees import trim_leaves
from LOTlib.Miscellaneous import None2Empty

class PartitionMCMC(MultipleChainMCMC):

    def __init__(self, grammar, make_h0, data, depth, increment_from=None, steps=Infinity, grammar_optimize=True, **kwargs):
        """
            Initializer.
            
            *grammar* - what grammar are we using?
            *make_h0* - a function to generate h0s. This MUST take a value argument to set the value
            *data*    - D for P(H|D)
            *depth*   - How deep do we enumerate?
            
            *grammar_optimize* -- if True, we trim the grammar by removing rules that lead to just terminals. This means all branches start with the same terminals, but these will be resampeld aways. Can be exponential speedup initialization
        """
        
        # make a version with only one terminal per nonterminal, or else we can get really exponential 
        # generation times
        if grammar_optimize:
            from copy import deepcopy
            grammar = deepcopy(grammar)
            for nt in grammar.rules.keys():
                found_terminal = False
                newrules = []
                for r in grammar.rules[nt]:
                    if any([ grammar.is_nonterminal(a) for a in None2Empty(r.to) ]):
                        newrules.append(r)
                    elif not found_terminal:
                        found_terminal = True
                        newrules.append(r)
                grammar.rules[nt] = newrules
        
        
        # first figure out how many trees we have
        # We generate a lot and then replace terminal nodes with their types, because the terminal nodes will
        # be the only ones that are allowed to be altered *within* a chain. So this collapses together
        # trees that differ only on terminals
        partitions = [] # the "Starting" trees for a partition -- includes an arbitrary terminal
        seen_collapsed = set() # what have we seen the collapsed forms of?
        for t in grammar.increment_tree(x=increment_from, depth=depth):
            ct = trim_leaves(t)
            
            if ct not in seen_collapsed:
                seen_collapsed.add(ct)
                partitions.append(t)
        
        #print "# Using partitions:", partitions
        
        # Take each partition (h0) and set it to have zero resample_p exact at the leaf
        for p in partitions:
            #print p
            for t in p:
                if not t.is_terminal():
                    t.resample_p = 0.0
                else:
                    t.resample_p = 1.0
        
        # initialize each chain
        MultipleChainMCMC.__init__(self, lambda: None, data, steps=steps, nchains=len(partitions), **kwargs)
        
        # And set each to the partition
        for c,p in zip(self.chains, partitions):
            c.set_state(make_h0(value=p))
            


if __name__ == "__main__":
    
    from LOTlib.Examples.Number.Shared import grammar, make_h0, generate_data
    data = generate_data(300)
    
    #from LOTlib.Examples.RationalRules.Shared import grammar, data, make_h0
    
    #PartitionMCMC(grammar, make_h0, data, 2, skip=0)
    for h in PartitionMCMC(grammar, make_h0, data, 0, skip=0):
        print h.posterior_score, h
    
 