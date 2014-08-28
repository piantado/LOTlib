"""
    The intuition here is that one way to prevent chains from duplicating work is to divide up the space and ensure chains stay 
    in separate regions. This is accomplished here by setting the resample_p to very tiny except for the leaves, after
    enumerating up to some depth in the grammar. Warning: the number of trees (and thus chains) will likely be exponential in the depth!
    
    This is at present NOT a correct sampler since it doesn't correctly handle proposals from one partition to another. It should be a correct sampler 
    within each partition.  

"""
from LOTlib.Miscellaneous import Infinity
from copy import copy


from MultipleChainMCMC import MultipleChainMCMC

from LOTlib.Subtrees import trim_leaves
from LOTlib.Miscellaneous import None2Empty

class PartitionMCMC(MultipleChainMCMC):

    def __init__(self, grammar, make_h0, data, max_depth=3, increment_from=None, yield_partition=False, steps=Infinity, grammar_optimize=True, **kwargs):
        """
            Initializer.
            
            *grammar* - what grammar are we using?
            *make_h0* - a function to generate h0s. This MUST take a value argument to set the value
            *data*    - D for P(H|D)
            *max_depth*, *max_n* -- only one of these may be specified. Either enumerate up to depth max_depth, or enumerate up to the largest depth such that the number of trees is less than max_N
            
            TODO: We can in principle optimize the grammar by including only one rule of the form NT->TERMiNAL for each NT. This will exponentially speed things up...
        """
        
        partitions = []
   
        # first figure out how many trees we have
        # We generate a lot and then replace terminal nodes with their types, because the terminal nodes will
        # be the only ones that are allowed to be altered *within* a chain. So this collapses together
        # trees that differ only on terminals
        seen_collapsed = set() # what have we seen the collapsed forms of?
        for t in grammar.increment_tree(x=increment_from, max_depth=max_depth):
            ct = trim_leaves(t)
            if ct not in seen_collapsed:
                seen_collapsed.add(ct)
                partitions.append(t)
        
    #print "# Using partitions:", partitions
        
        # Take each partition (h0) and set it to have zero resample_p exact at the leaf
        for p in partitions:
            print p
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
        
        # and store these
        self.partitions = map(copy, partitions)


if __name__ == "__main__":
    
    from LOTlib.Examples.Number.Shared import grammar, make_h0, generate_data
    data = generate_data(300)
    
    #from LOTlib.Examples.RegularExpression.Shared import grammar, make_h0, data
        
    #from LOTlib.Examples.RationalRules.Shared import grammar, data, make_h0
    
    #PartitionMCMC(grammar, make_h0, data, 2, skip=0)
    for h in PartitionMCMC(grammar, make_h0, data, max_N=100, skip=0):
        print h.posterior_score, h
        break
    
 