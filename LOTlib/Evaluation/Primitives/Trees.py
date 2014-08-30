from Primitives import LOTlib_primitive
from LOTlib.FunctionNode import FunctionNode, isFunctionNode

import re ## TODO: WHY? PROBABLY BAD FORM

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Tree operations
# In a tree T, check relations between some elements. Sometimes T is
# not used, but we leave it in all functions for simplicity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@LOTlib_primitive
def is_(x,y): return (x is y)

@LOTlib_primitive
def equals_(x,y): return (x == y)

@LOTlib_primitive
def co_referents_(T,x):
    """
            The co-referents of x in t
    """
    return filter(lambda si: co_refers(si,x), T)

@LOTlib_primitive
def sisters_(T, x, y, equality=False):
    """
            Check if x,y are sisters in T
    """
    for s in T:
        if immediately_dominates(s,x) and immediately_dominates(s,y): return True
    return False

#@ We define a non-LOTlib version so we can use this in other definitions
def immediately_dominates(x, y):
    if isinstance(x, FunctionNode):
        for s in x.args:
            if s is y: return True

    return False

@LOTlib_primitive
def immediately_dominates_(x, y):
    return immediately_dominates(x,y)

@LOTlib_primitive
def dominates_(x,y):
    """
            This checks if x >> y, but using object identity ("is") rather than equality
    """
    if x is y: return False # A node does not dominate itself
    if isinstance(x, FunctionNode):
        for s in x:
            if s is y: return True
    return False

@LOTlib_primitive
def tree_up_(T,x): return tree_up(T,x)

#@Define a non-LOTlib version for defining others
def tree_up(T, x):
    """
            Go up one node in the tree. If you are root, return None

            NOTE: Super slow since we search over the whole tree each time.... This is a little tricky otherwise without pointers
    """

    if x is T: return None

    for s in T:
        if immediately_dominates(s,x): return s
    return None


@LOTlib_primitive
def children_(x):
    if isinstance(x, FunctionNode): return [ c for c in x.args ]
    else: return []

@LOTlib_primitive
def descendants_(x):
    if isinstance(x, FunctionNode): return [ c for c in x ]
    else: return []

@LOTlib_primitive
def tree_root_(T):
    return T

@LOTlib_primitive
def is_nonterminal_type_(x,y): return is_nonterminal_type(x,y)

no_coref_regex = re.compile(r"\..+$")
def is_nonterminal_type(x,y):
    # Check if x is of a given type, but remove corefence information from X (y is the type)

    if x is None or y is None: return False
    if isinstance(x,list): return False # a list can't be a nonterminal

    if not isinstance(x,str): x = x.name

    # remove the .coreference info
    x = no_coref_regex.sub("", x)

    return (x==y)

@LOTlib_primitive
def ancestors_(T, x): return ancestors(T,x)

#def ancestors(T,x):
    ### SLOW VERSION -- O(N^2) since tree_up is O(N)
    #if not isinstance(x, FunctionNode): return []
    #out = []
    #while not tree_is_(x,T):
        #x = tree_up(T,x)
        #out.append(x)
    #return out

def ancestors(T,x):
    """
            Here is a version of ancestors that is O(n), rather than the repeated calls to tree_up, which is O(N^2)
    """

    anc = []

    def recurse_down(y):
        #print "RD", y, "\t", x
        if isinstance(y,list):
            return any(map(recurse_down, filter(isFunctionNode, y)))
        elif isFunctionNode(y):
            if recurse_down(y.args) or immediately_dominates(y, x):
                anc.append(y) # put y on the end
                return True
            return False

    recurse_down(T)

    return anc

@LOTlib_primitive
def whole_tree_(T):
    # LIST type of all elements of T
    return [ti for ti in T ]

@LOTlib_primitive
def tree_is_(x,y): return (x is y)


@LOTlib_primitive
def co_refers_(x,y): return co_refers(x,y)

coref_matcher = re.compile(r".+\.([0-9]+)$") ## Co-reference (via strings)
def co_refers(x,y):

    if x is y: return False # By stipulation, nothing co-refers to itself

    # Weird corner cases
    if isinstance(x,list) or isinstance(y,list): return False
    if x is None or y is None: return False

    ## Check if two FunctionNodes or strings co-refer (e.g. are indexed with the same .i in their name)
    xx = x.name if isFunctionNode(x) else x
    yy = y.name if isFunctionNode(y) else y

    mx = coref_matcher.search(xx)
    my = coref_matcher.search(yy)

    if mx is None or my is None:
        return False
    else:
        return (mx.groups("X")[0] == my.groups("Y")[0]) # set the default in groups so that they won't be equal if there is no match

@LOTlib_primitive
def non_xes_(x,T):
    return filter(lambda v: v is not x, T)

@LOTlib_primitive
def first_dominating_(T,x,t):
    # Returns the first thing dominating x of type t
    # And None otherwise

    if isFunctionNode(x):
        up = tree_up(T,x)
        while up is not None:
            if is_nonterminal_type(up,t): return up
            up = tree_up(T,up)

    return None

@LOTlib_primitive
def first_dominated_(x,t):
    # Returns the first thing dominating x of type t
    # And None otherwise
    if isFunctionNode(x):
        for sn in x:
            if is_nonterminal_type(sn, t): return sn

    return None
