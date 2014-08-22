"""
    A simple class to do inference via enumeration
"""
from copy import copy

class EnumerationInference(object):
    
    def __init__(self, grammar, make_h, data, increment_from=None, max_depth=15):
        """
            make_h should take a value (a tree) and return something of type Hypothesis
        """ 
        
        self.__dict__.update(locals())
        
    def __iter__(self):
        
        for t in self.grammar.increment_tree(x=self.increment_from, max_depth=self.max_depth):
            h = self.make_h(t)
            h.compute_posterior(self.data)
            yield h
            

if __name__ == "__main__":
    
    
    #from LOTlib.Examples.Number.Shared import grammar, make_h0, generate_data
    #data = generate_data(300)
    
    from LOTlib.Examples.RegularExpression.Shared import grammar, make_h0, data
        
    #from LOTlib.Examples.RationalRules.Shared import grammar, data, make_h0
    
    #PartitionMCMC(grammar, make_h0, data, 2, skip=0)
    for h in EnumerationInference(grammar, make_h0, data, max_depth=7):
        print h.posterior_score, h
    
                
                
        
    