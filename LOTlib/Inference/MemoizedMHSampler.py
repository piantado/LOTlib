
from LOTlib.Miscellaneous import q, Infinity
from MetropolisHastings import MHSampler
from cachetools import LRUCache

class MemoizedMHSampler(MHSampler):
    """
        Same as MHSampler, but the values of compute_posterior are cached via LRUCache
    """
    def __init__(self, h0,  data, memoize=Infinity, **kwargs):
        MHSampler.__init__(self, h0, data, **kwargs)
        
        # self.mem stores return of compute_posterior
        self.mem = LRUCache(maxsize=memoize)
        
    def compute_posterior(self, h, data):
        if h in self.mem:
            ret = self.mem[h]
            h.posterior_score = sum(ret) # set this because it may not be set
            return ret
        else:
            ret = MHSampler.compute_posterior(self, h, data)
            self.mem[h] = ret
            return ret

if __name__ == "__main__":
    
    from LOTlib.Examples.Number.Shared import generate_data, NumberExpression, grammar, get_knower_pattern
    
    data = generate_data(100)
    h0 = NumberExpression(grammar)    
    sampler = MemoizedMHSampler(h0, data, steps=1000)
    for h in sampler:
        pass #print q(get_knower_pattern(h)), h.posterior_score, h.prior, h.likelihood, q(h), sampler.acceptance_count, sampler.acceptance_ratio()
    
