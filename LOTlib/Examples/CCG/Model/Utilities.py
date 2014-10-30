import re
from LOTlib.Hypotheses.LOTHypothesis import LOTHypothesis
from Grammar import grammar

# What are the objects we may use?
OBJECTS              = ['JOHN', 'MARY', 'SUSAN', 'BILL']
SEMANTIC_1PREDICATES = ['SMILED', 'LAUGHED', 'MAN', 'WOMAN']
SEMANTIC_2PREDICATES = ['SAW', 'LOVED']

def str2sen(s):
    # Chop up a string by spaces to make a "Sentence"
    return re.split(r'\s', s)

def can_compose(a,b):
    """
            Takes two TYPES, and returns the result of a(b)
            IF this is not possible (due to the types), return None.

            NOTE: No currying, type-raising or anything fancy (yet)
    """

    # We can't compose if a is not a function (it's type is not a list)
    if not isinstance(a, tuple): ## TODO: NOTE THAT WE don't allow other iterables than tuples (not even lists)
        return None
    else:
        ato, afrom = a

        if afrom == b: return ato
        else:          return None


# How we make a hypothesis inside the lexicon
def make_hypothesis():
    return LOTHypothesis(grammar, args=['C'])
