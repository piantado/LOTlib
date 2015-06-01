import re

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

