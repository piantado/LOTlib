"""
    A version of grammar that allows nodes to be expanded to real values
    Very experimental.
"""

from LOTlib.Grammar import Grammar
from LOTlib.FunctionNode import FunctionNode

from LOTlib.Miscellaneous import normlogpdf

try:                import numpy as np
except ImportError: import numpypy as np

# The resample probability for constants
# TODO: Should probably be in Grammar?
CONSTANT_RESAMPLE_P = 1.0

class RealValueGrammar(Grammar):

    def generate(self, x='*USE_START*', d=0):
        """
              RealValueGrammar.generate may create gaussians or uniforms when given "*gaussian*" and "*uniform*" as the nonterminal type.
              Otherwise, this is identical to LOTlib.Grammar
        """
            if x == '*USE_START*':
                x = self.start

            if x=='*gaussian*':
                # TODO: HIGHLY EXPERIMENTAL!!
                # Wow this is really terrible for mixing...
                v = np.random.normal()
                gp = normlogpdf(v, 0.0, 1.0)
                return FunctionNode(returntype=x, name=str(v), args=None, generation_probability=gp, ruleid=0, resample_p=CONSTANT_RESAMPLE_P ) ##TODO: FIX THE ruleid

            elif x=='*uniform*':
                v = np.random.rand()
                gp = 0.0
                return FunctionNode(returntype=x, name=str(v), args=None, generation_probability=gp, ruleid=0, resample_p=CONSTANT_RESAMPLE_P ) ##TODO: FIX THE ruleid
            else:
                # Else call normal generation
                Grammar.generate(self,x,d=d)
