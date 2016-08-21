"""
    A simple class to do inference via enumeration
"""

from LOTlib.Miscellaneous import Infinity, self_update

class EnumerationInference(object):
    
    def __init__(self, grammar, make_h, data, steps=Infinity):
        self_update(self, locals())
        
    def __iter__(self):
        for i, t in enumerate(self.grammar.enumerate()):

            if i >= self.steps:
                raise StopIteration

            h = self.make_h(value=t)
            h.compute_posterior(self.data)
            yield h

            

if __name__ == "__main__":
    
    from LOTlib import break_ctrlc

    #from LOTlib.Examples.Number.Shared import grammar, make_h0, generate_data
    #data = generate_data(100)
    #from LOTlib.Examples.RegularExpression.Shared import grammar, make_h0, data

    from LOTlib.Examples.Magnetism.Simple import make_data, make_hypothesis
    from LOTlib.Examples.Magnetism.Simple.Grammar import grammar


    #from LOTlib.Examples.RationalRules.Shared import grammar, data, make_h0

    for h in break_ctrlc(EnumerationInference(grammar, make_hypothesis, make_data(), steps=10000)):
        print h.posterior_score, h
    
                
                
        
    