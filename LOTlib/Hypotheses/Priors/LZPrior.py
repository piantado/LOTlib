
# Two possible libraries
# import lzw # needed for LZ compression
# import lzstring

from LOTlib.Miscellaneous import attrmem

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

        # first pack up
        s = self.grammar.pack_ascii(self.value)

        # then compress with lzw
        c = compressor.compressToBase64(s)

        # then take the compressed length
        # NOTE: This length will probability only be accurate
        # to within a char, or 8 bits
        return -(len(c)-minL) / self.prior_temperature