"""
    Multiple parallel chains running at once. 
    
    This runs skip steps within each chain, looping to return samples in a roundrobin fashion,
    chain1, chain2, chain3, ...
    
    Also, steps is the *total* number of steps, not the number of steps for each chain
    This is subclassed by several other inference techniques
"""
from LOTlib.Miscellaneous import Infinity
from MetropolisHastings import MHSampler
from numpy import mean


class MultipleChainMCMC(object):
    def __init__(self, make_h0, data, steps=Infinity, chains=10, **kwargs):
        
        self.nchains = chains
        self.chain_idx = -1 # what chain are we on? This get incremented before anything, so it starts with 0
        
        self.chains = [ MHSampler( make_h0(), data, steps=steps/chains, **kwargs) for _ in xrange(chains) ]
    
    def __iter__(self):
        return self
    
    def next(self):
        self.chain_idx = (self.chain_idx+1)%self.nchains
        return self.chains[self.chain_idx].next()
        
    def reset_counters(self):
        for c in self.chains:
            c.reset_counters()
    
    def acceptance_ratio(self):
        """
            Return the mean acceptance rate of all chains
        """
        return mean([c.acceptance_ratio() for c in self.chains])
    
if __name__ == "__main__":
    from LOTlib.Examples.Number.Shared import generate_data, NumberExpression, grammar
    data = generate_data(300)
    
    make_h0 = lambda : NumberExpression(grammar)
    
    sampler = MultipleChainMCMC(make_h0, data)
    for h in sampler:
        print h.posterior_score, h
        
      
    
                       