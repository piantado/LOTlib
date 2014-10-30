"""
    A simple class to do inference via enumeration
"""

from LOTlib.Miscellaneous import Infinity

class EnumerationInference(object):
    
    def __init__(self, grammar, make_h, data, steps=Infinity):
        self.__dict__.update(locals())
        
    def __iter__(self):
        for i, t in enumerate(self.grammar.enumerate()):

            if i > self.steps:
                break

            h = self.make_h(value=t)
            h.compute_posterior(self.data)
            yield h

            

if __name__ == "__main__":
    
    from LOTlib import lot_iter

    #from LOTlib.Examples.Number.Shared import grammar, make_h0, generate_data
    #data = generate_data(100)
    #from LOTlib.Examples.RegularExpression.Shared import grammar, make_h0, data

    from LOTlib.Examples.Magnetism.Simple.Run import grammar, data, make_h0

    #from LOTlib.Examples.RationalRules.Shared import grammar, data, make_h0

    for h in lot_iter(EnumerationInference(grammar, make_h0, data)):
        print h.posterior_score, h
    
                
                
        
    