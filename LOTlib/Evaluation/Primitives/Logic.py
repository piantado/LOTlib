from Primitives import LOTlib_primitive

import itertools

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic logic
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
def id_(A): return A # an identity function

@LOTlib_primitive
def and_(A,B): return (A and B)

@LOTlib_primitive
def AandnotB_(A,B): return (A and (not B))

@LOTlib_primitive
def notAandB_(A,B): return ((not A) and B)

@LOTlib_primitive
def AornotB_(A,B): return (A or (not B))

@LOTlib_primitive
def A_(A,B): return A

@LOTlib_primitive
def notA_(A,B): return not A

@LOTlib_primitive
def B_(A,B): return B

@LOTlib_primitive
def notB_(A,B): return not B

@LOTlib_primitive
def nand_(A,B): return not (A and B)

@LOTlib_primitive
def or_(A,B): return (A or B)

@LOTlib_primitive
def nor_(A,B): return not (A or B)

@LOTlib_primitive
def xor_(A,B): return (A and (not B)) or ((not A) and B)

@LOTlib_primitive
def not_(A): return (not A)

@LOTlib_primitive
def implies_(A,B): return (A or (not B))

@LOTlib_primitive
def iff_(A,B): return ((A and B) or ((not A) and (not B)))

@LOTlib_primitive
def if_(C,X,Y):
    if C: return X
    else: return Y

@LOTlib_primitive
def gt_(x,y): return x>y

@LOTlib_primitive
def gte_(x,y): return x>=y

@LOTlib_primitive
def lt_(x,y): return x<y

@LOTlib_primitive
def lte_(x,y): return x<=y

@LOTlib_primitive
def eq_(x,y): return x==y

@LOTlib_primitive
def zero_(x,y): return x==0.0


@LOTlib_primitive
def streq_(x,y): return str(x)==str(y)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Quantification
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
def not_exists_(F,S): return not exists_(F,S)

@LOTlib_primitive
def exists_(F,S): return exists(F,S)
def exists(F,S):
    return any(itertools.imap(F,S)) # This appears to be faster than writing it ourself

@LOTlib_primitive
def not_forall_(F,S): return not forall(F,S)

@LOTlib_primitive
def forall_(F,S): return forall(F,S)

def forall(F,S):
    return all(itertools.imap(F,S))

@LOTlib_primitive
def iota_(F,S):
    """
        The unique F in S. If none, or not unique, return None
    """
    match = None
    for s in S:
        if F(s):
            if match is None: match = s
            else: return None  # We matched more than one
    return match


