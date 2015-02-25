"""
    The intuition here is that one way to prevent chains from duplicating work is to divide up the space and ensure chains stay 
    in separate regions. This is accomplished here by setting the resample_p to very tiny except for the leaves, after
    enumerating up to some depth in the grammar. Warning: the number of trees (and thus chains) will likely be exponential in the depth!
    
    This is at present NOT a correct sampler since it doesn't correctly handle proposals from one partition to another. It should be a correct sampler 
    within each partition.  

"""
from LOTlib import lot_iter
from LOTlib.Miscellaneous import Infinity, infrange
from copy import copy


from MultipleChainMCMC import MultipleChainMCMC

from LOTlib.Subtrees import trim_leaves
from LOTlib.Miscellaneous import None2Empty, lambdaNone
from LOTlib.Inference.Proposals.RegenerationProposal import RegenerationProposal

class BreakException(Exception):
    """
    Break out of multiple loops
    """
    pass

# Now we need to define a class to wrap in resample_p so it gets used here.
class MyProposal(RegenerationProposal):
    def propose_tree(self, t):
        return RegenerationProposal.propose_tree(self, t, resampleProbability=lambda x: getattr(x,'resample_p', 1.0))

class PartitionMCMC(MultipleChainMCMC):
    """
        Partition-based MCMC.

        Note that it currently only works with RegenerationProposal

    """

    def __init__(self, grammar, make_h0, data, max_N=1000, steps=Infinity, **kwargs):
        """
        :param grammar: -- the grammar we use
        :param make_h0: -- make a hypothesis
        :param data:    -- data for the posterior
        :param max_N: -- the max number of samples we'll take
        :param steps: -- how many steps
        :return:
        """

        # first figure out the depth we can go to without exceeding max_N
        partitions = []
        try:
            for d in infrange():
                #print "# trying ", d
                tmp = []
                for i, t in enumerate(grammar.enumerate_at_depth(d, leaves=False)):
                    tmp.append(t)
                    if i > max_N:
                        raise BreakException

                # this gets set if we successfully exit the loop
                # so it will store the last set that didn't exceed size max_N
                partitions = tmp
        except BreakException:
            pass

        assert len(partitions) > 0

        # and store these so we can see them later
        self.partitions = map(copy, partitions)

        # Take each partition, which doesn't have leaves, and generate leaves, setting
        # it to a random generation (fill in the leaves with random hypotheses)
        for p in partitions:

            print "# Initializing partition:", p

            for n in p.subnodes():
                # set to not resample these
                setattr(n, 'resample_p', 0.0) ## NOTE: This is an old version of how proposals were made, but we use it here to store in each node a prob of being resampled

                # and fill in the missing leaves with a random generation
                for i, a in enumerate(n.args):
                    if grammar.is_nonterminal(a):
                        n.args[i] = grammar.generate(a)
        print "# Initialized %s partitions" % len(partitions)

        # initialize each chain
        MultipleChainMCMC.__init__(self, lambdaNone, data, steps=steps, nchains=len(partitions), **kwargs)


        # And set each to the partition
        for c, p in zip(self.chains, partitions):
            # We need to make sure the proposal_function is set to use resample_p, which is set above
            v = make_h0(value=p, proposal_function=MyProposal(grammar) )
            c.set_state(v, compute_posterior=False) # and make v use the right proposal function


    def current_partition(self):
        """
        Which partition did the sample just come from?
        :return:
        """
        return self.partitions[self.chain_idx]


if __name__ == "__main__":
    
    from LOTlib.Examples.Number.Model import grammar, make_h0, generate_data
    data = generate_data(300)
    
    #from LOTlib.Examples.RegularExpression.Shared import grammar, make_h0, data
    #from LOTlib.Examples.RationalRules.Shared import grammar, data, make_h0

    pmc = PartitionMCMC(grammar, make_h0, data, max_N=10, skip=0)
    for h in lot_iter(pmc):
        print h.posterior_score, pmc.current_partition(), "\t", h

    
 