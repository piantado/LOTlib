from math import log
from LOTlib.Hypotheses.SimpleLexicon import SimpleLexicon

class MutuallyRecursiveLexicon(SimpleLexicon):
    """
    This is just a SimpleLexicon whose __call__ functiond defaultly passes "self" as the first argument. This means
    that the word values must take "self", but it also means they can call "self" (called "lexicon") internally
    to define words in terms of each other.
    """

    def __call__(self, word, *args):
        """
        Wrap in self as a first argument that we don't have to in the grammar. This way, we can use self(word, number) as above.
        """
        return self.value[word](self, *args)  # pass in "self" as lex

    def compute_single_likelihood(self, datum, ALPHA=0.9):
        ret = None
        try:  # Must catch errors here and NOT in __call__, since doing so in call will allow us to loop infinitely
            ret = self(*datum.input)
        except RuntimeError as e: # we get this from recursing too deep -- catch and thus treat "ret" as None
            pass

        return log(ALPHA*(ret == datum.output) + (1.-ALPHA)/2.)