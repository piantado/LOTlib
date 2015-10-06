from Model import *

from Model.Grammar import lot_grammar as grammar

h0 = NumberGameHypothesis(grammar=grammar)

data = [FunctionData(input=[], output=[2, 8, 9, 10, 16], alpha=0.99)]

from LOTlib import break_ctrlc
from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.SampleStream import *

for h in SampleStream(break_ctrlc(MHSampler(h0, data))) >> Tee(Skip(100) >> PosteriorTrace(), Unique() >> PrintH()):
    print "#", h()
