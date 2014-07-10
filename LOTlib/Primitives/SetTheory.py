from Primitives import LOTlib_primitive

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set-theoretic primitives
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@LOTlib_primitive
def set_(*args): return set(args)

@LOTlib_primitive
def set_add_(x,s):
    s.add(x)
    return s

@LOTlib_primitive
def union_(A,B): return A.union(B)

@LOTlib_primitive
def intersection_(A,B): return A.intersection(B)

@LOTlib_primitive
def setdifference_(A,B): return A.difference(B)

@LOTlib_primitive
def select_(A): # choose an element, but don't remove it
    
    try: # quick selecting without copying
        return set([iter(A).next()])
    except StopIteration:
        return set()
    
    #if len(A) > 0:
        #x = A.pop()
        #A.add(x)
        #return set([x]) # but return a set
    #else: return set() # empty set

from random import sample as random_sample
@LOTlib_primitive
def sample_unique_(S):
    return random_sample(S,1)[0]

from random import choice as random_choice
@LOTlib_primitive
def sample_(S):
    if len(S) == 0: return set()
    else:           return random_choice(list(S))


@LOTlib_primitive
def exhaustive_(A,B): return coextensive(A,B)

@LOTlib_primitive
def coextensive_(A,B): return coextensive(A,B)
def coextensive(A,B): # are the two sets coextensive?
    #print A,B
    return (A.issubset(B) and B.issubset(A))

@LOTlib_primitive
def equal_(A,B): return (A == B)

@LOTlib_primitive
def equal_word_(A,B): return (A == B)

@LOTlib_primitive
def empty_(A): return (len(A)==0)

@LOTlib_primitive
def nonempty_(A): return not empty_(A)

@LOTlib_primitive
def cardinality1_(A): return (len(A)==1)

@LOTlib_primitive
def cardinality2_(A): return (len(A)==2)

@LOTlib_primitive
def cardinality3_(A): return (len(A)==3)

@LOTlib_primitive
def cardinality4_(A): return (len(A)==4)

@LOTlib_primitive
def cardinality5_(A): return (len(A)==5)

@LOTlib_primitive
def cardinality_(A): return len(A)

# returns cardinalities of sets and otherwise numbers -- for duck typing sets/ints
def cardify(x):
    if isinstance(x, set): return len(x)
    else: return x

@LOTlib_primitive
def cardinalityeq_(A,B): return cardify(A) == cardify(B)

@LOTlib_primitive
def cardinalitygt_(A,B): return cardify(A) > cardify(B)

@LOTlib_primitive
def cardinalitylt_(A,B): return cardify(A) > cardify(B)

@LOTlib_primitive
def subset_(A,B):
    return A.issubset(B)

@LOTlib_primitive
def is_in_(x,S):
    return (x in S)
