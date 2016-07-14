from LOTlib.Miscellaneous import attrmem,Infinity
from LOTlib.Grammar import pack_string
from LZutil.IntegerCodes import to_fibonacci as integer2bits # Use Mackay's Fibonacci code
from LZutil.LZ2 import encode

class LZPrior(object):
    """
    A prior that is based on using Lempel-Ziv on the tree expansions.
    Prior is proportional to 2^(-length)
    """

    @attrmem('prior')
    def compute_prior(self):
        if self.value.count_subnodes() > getattr(self, 'maxnodes', Infinity):

            return -Infinity

        s = self.grammar.pack_ascii(self.value)
        # 1+ since it must be positive
        bits = ''.join([ integer2bits(1+pack_string.index(x)) for x in s ])
        c = encode(bits, pretty=0)

        return -len(c)
