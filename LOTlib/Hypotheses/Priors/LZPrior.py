
# Two possible libraries
# import lzw # needed for LZ compression
# import lzstring

from LOTlib.Miscellaneous import attrmem
from LOTlib.Grammar import pack_string

from LZutil.IntegerCodes import to_fibonacci as integer2bits # Use Mackay's Fibonacci code
from LZutil.LZ2 import encode



# compressor = lzstring.LZString()
# # let's subtract off the min possible for some one character string
# minL = len(lzstring.LZString().compressToBase64("a"))

class LZPrior(object):
    """
    A prior that is based on using Lempel-Ziv on the tree expansions.
    Prior is proportional to 2^(-length)
    """

    @attrmem('prior')
    def compute_prior(self):
        s = self.grammar.pack_ascii(self.value)
        bits = ''.join([integer2bits(1+pack_string.index(x)) for x in s ]) # 1+ since it must be positive
        c = encode(bits, pretty=0)

        return -len(c)

        # # first pack up
        # s = self.grammar.pack_ascii(self.value)
        #
        # # then compress with lzw
        # c = compressor.compressToBase64(s)
        #
        # # then take the compressed length
        # # NOTE: This length will probability only be accurate
        # # to within a char, or 8 bits
        # return -(len(c)-minL) / self.prior_temperature