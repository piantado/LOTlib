"""
Routines for combinatory logic evaluation.

This provides an explicit evaluator for a sequence of combinators like ['I' 'S' ['K' 'S' 'S']]
which maps to a new lists. This is slower than implementing them directly as functions,
but lets us see what they reduce to.

Relevant functions in Miscellaneous are:
        - LOTlib.Miscellaneous.unlist_singleton  (removes extraneous lists)
        - LOTlib.Miscellaneous.list2sexpstr      (converts a list of lists into an S-expression string)

Much much faster scheme code is available from Steve.

"""

from LOTlib.Miscellaneous import unlist_singleton
from LOTlib.Eval import EvaluationException
MAX_DEPTH = 25
MAX_LENGTH = 25

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# exceptions for combinators

class CombinatorEvaluationException(EvaluationException): pass

class CombinatorEvaluationDepthException(CombinatorEvaluationException): pass

class CombinatorEvaluationLengthException(CombinatorEvaluationException): pass

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# evaluate combinators

def combinator_reduce(lst, depth=MAX_DEPTH):
    """Reduce combinatory logic list of lists.

    TODO: Incorporate other reduction strategies / evaluation orders.

    """

    if not isinstance(lst, list): return lst
    elif len(lst) == 0: return list()
    elif depth < 0: raise CombinatorEvaluationDepthException
    elif len(lst) > MAX_LENGTH: raise CombinatorEvaluationLengthException
    else:
        op, args = lst[0], lst[1:]
        newdepth = depth-1

        if isinstance(op, list):
            return combinator_reduce( op + args, newdepth )
        elif op == 'I' and len(args) >= 1:
            return combinator_reduce( args[1:], newdepth)
        elif op == 'K' and len(args) >= 2:
            x,y,rest = args[0], args[1], args[2:]
            return combinator_reduce( [x] + rest, newdepth)
        elif op == 'S' and len(args) >= 3:
            x,y,z,rest = args[0], args[1], args[2], args[3:]
            return combinator_reduce( [x, z, [y, z]] + rest, newdepth )
        else:
            rest = map(lambda x: unlist_singleton(combinator_reduce(x, newdepth)), args)

            if len(rest) == 0: return lst
            else:              return [op, ] + rest

if __name__ == "__main__":

    from LOTlib.Parsing import parseScheme

    print combinator_reduce( parseScheme("(S (K (S I)) (S (K K) I) x y)"))

    print combinator_reduce( parseScheme("(S (K (S I))  (S (K K) I) x x  )"))

    print combinator_reduce( parseScheme("(S (S x) x)")      )

    print combinator_reduce( parseScheme("(K (S (S x) x) y z)")      )

    print combinator_reduce( parseScheme("(S (I (I (I))))")  )
