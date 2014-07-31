from Primitives import LOTlib_primitive

import math
from numpy import sign

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Constants
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        
PI = math.pi
TAU = 2.0*PI
E = math.e

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic arithmetic
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@LOTlib_primitive
def negative_(x): return -x
def neg_(x): return -x

@LOTlib_primitive
def plus_(x,y): return x+y

@LOTlib_primitive
def times_(x,y): return x*y

@LOTlib_primitive
def divide_(x,y): 
    if y != 0.: return x/y
    else:     return float("inf")*x
    
@LOTlib_primitive
def subtract_(x,y): return x-y
    
@LOTlib_primitive
def minus_(x,y): return x-y

@LOTlib_primitive
def sin_(x): 
    try:
        return math.sin(x)
    except: return float("nan")

@LOTlib_primitive
def cos_(x): 
    try:
        return math.cos(x)
    except: return float("nan")

@LOTlib_primitive
def tan_(x): 
    try:
        return math.tan(x)
    except: return float("nan")

@LOTlib_primitive    
def sqrt_(x): 
    try: return math.sqrt(x)
    except: return float("nan")

@LOTlib_primitive
def pow_(x,y): 
    #print x,y
    try: return pow(x,y)
    except: return float("nan")

@LOTlib_primitive
def abspow_(x,y): 
    """ Absolute power. sign(x)*abs(x)**y """
    #print x,y
    try: return sign(x)*pow(abs(x),y)
    except: return float("nan")

@LOTlib_primitive
def exp_(x): 
    try: 
        r = math.exp(x)
        return r
    except: 
        return float("inf")*x

@LOTlib_primitive
def abs_(x): 
    try: 
        r = abs(x)
        return r
    except: 
        return float("inf")*x
        
        
@LOTlib_primitive    
def log_(x): 
    if x > 0: return math.log(x)
    else: return -float("inf")

@LOTlib_primitive
def log2_(x): 
    if x > 0: return math.log(x)/math.log(2.0)
    else: return -float("inf")
    
@LOTlib_primitive
def pow2_(x): 
    return math.pow(2.0,x)

@LOTlib_primitive
def mod_(x,y):
    if y==0.0: return float("nan")
    return (x%y)
