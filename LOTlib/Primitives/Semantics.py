from LOTlib.Eval import primitive

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For language / semantics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@primitive
def presup_(a,b):
    if a: return b
    else:
        if b: return "undefT" # distinguish these so that we can get presup out
        else: return "undefF"

@primitive
def is_undef_(x):
    return is_undef(x)

def is_undef(x):
    if isinstance(x,list):
        return map(is_undef, x)
    else:
        return (x is None) or (x =="undefT") or (x == "undefF") or (x == "undef")

@primitive
def collapse_undef(x):
    """
        Change undefT->True and undefF->False
    """
    if isinstance(x,list): return map(collapse_undef, x)
    else:
        if    x is "undefT": return True
        elif  x is "undefF": return False
        else: x
