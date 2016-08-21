"""
    Operations on combinantory logic. There are two families of evaluation: either
    reduce will operate on strings "I", "S", "K" and list structures built of them,
    or the primitives I_, S_, K_ are callable primitives to implement these.

"""
from math import floor

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Evaluation of combinatory logic expressions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CombinatorReduceException(Exception):
    pass

def mycons(x,y):
    # Just append but make sure arguments are lists
    if not isinstance(x,list):
        x = [x]
    if not isinstance(y, list):
        y = [y]
    return x+y

def reduce(lst, maxsteps=50):
    """
        One step of reduction on a list of lists representation for CL
        (i.e. not cons, cdr, car version)
    """
    # print "\t", lst

    if maxsteps <= 0:
        raise CombinatorReduceException

    ln = len(lst)
    first = lst[0]

    if isinstance(first, list):
        return reduce(mycons(first,lst[1:]), maxsteps=maxsteps-1) # unlist the first argument
    elif first == "I" and ln >= 2:
        return reduce(lst[1:], maxsteps=maxsteps-1)
    elif first == "K" and ln >= 3:
        return reduce( mycons(lst[2], lst[3:]), maxsteps=maxsteps-1)
    elif first == "S" and ln >= 4:
        x,y,z = lst[1], lst[2], lst[3]
        return reduce( mycons([[x, z],[y, z]], lst[4:]), maxsteps=maxsteps-1)
    else:
        return mycons(first, [reduce(x, maxsteps=floor(maxsteps/len(lst))) for x in lst[1:]])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A grammar for simple CL expressions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from LOTlib.Grammar import Grammar
grammar = Grammar(start="CLEXPR")

# flattern2str lives at the top, and it takes a cons, cdr, car structure and projects it to a string
grammar.add_rule('CLEXPR', '[%s, %s]', ['CLEXPR', 'CLEXPR'], 1.0)

grammar.add_rule('CLEXPR', '"I"', None, 1.0)
grammar.add_rule('CLEXPR', '"S"', None, 1.0)
grammar.add_rule('CLEXPR', '"K"', None, 1.0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Just look a little
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    import LOTlib

    while not LOTlib.SIG_INTERRUPTED:
        x = eval(str(grammar.generate()))
        print x
        try:
            print reduce(x)
        except CombinatorReduceException:
            print "NON-HALT"