#!/usr/bin/env python
"""
Lempel-Ziv code for compression

This simple version of Lempel-Ziv does no dictionary management and
is inefficient in several ways. See MacKay (2003) Chapter 6. page 119

http://www.aims.ac.za/~mackay/itila/

  LZ.py is free software (c) David MacKay December 2005. License: GPL
"""
## For license statement see  http://www.gnu.org/copyleft/gpl.html

from IntegerCodes import *

verbose=0
class node:
    def __init__(self,substring,pointer):
        self.child = []
        self.substring = substring
        self.pointer = pointer
        if verbose:
            print pointer, " is pointer for ", `substring`
        pass

def find(f, seq):
    """Return first item in sequence where f(item) == True."""
    for item in seq:
        if f(item): 
            return item
    return None

def encode ( c, pretty=1 ): ## c is a list of characters (0/1) ; p is whether to print out prettily
    """
    Encode using Lempel-Ziv, as in MacKay (2003) Chapter 6. page 119
    Pretty printing
    >>> print encode(list("000000000000100000000000"),1)
    (,0)(1,0)(10,0)(11,0)(010,1)(100,0)(110,0)

    Normal printing
    >>> print encode(list("000000000000100000000000"),0)
    010100110010110001100
    """
    output =[]
    
    #initialize dictionary
    dic = [] 
    dic.append( node("",0) )
    
    while(len(c)>0): # point G
        substring = "";         latest = -1
        # read bits from c until we DON'T get a match with the dictionary
        while(len(c)>=0):
            ans = find( lambda p: p.substring == substring , dic )
            if ((ans == None) or (len(c)==0)):
                assert latest != -1 ## we should have gone round this loop once already
                # print out prevanswer's pointer and latest bit
                digits = ceillog ( len(dic) )
                output.append( printout( dec_to_bin( prevanswer.pointer , digits ) , latest ,pretty) )
                # append new string to dictionary
                dic.append( node(substring, len(dic) ) )
                break # go back to G
            else:
                prevanswer = ans
                latest=c.pop(0)
                substring = substring+latest
            pass
        pass
    return "".join(output)

def printout( pointerstring, latest , pretty=1):
    if pretty:
        return "("+pointerstring+","+latest+")"
    else:
        return pointerstring+latest

def decode( c ):
    """
    >>> print decode(list("100011101100001000010"))
    1011010100010
    """
    output = [] 
    #initialize dictionary
    dic = [] 
    dic.append( node("",0) )
    while(len(c)>0):
        digits = ceillog ( len(dic) )
        pointer = bin_to_dec( c , digits )
        # find the dictionary entry with that pointer
        ans = find( lambda p: p.pointer == pointer , dic )
        substring = ans.substring
        output.append( substring )
        if (len(c)>0):
            latest=c.pop(0)
            output.append( latest )
            substring = substring+latest
            dic.append( node(substring, len(dic) ) )
            pass
        pass
    return "".join(output)
            
def test():
    print "encoding examples:"
    examples = [ "101101010001000000", "000000000000100000000000" ]            
    for ex in examples :
        print ex, encode( list(ex) )

    print "decoding examples:"
    examples = [ "100011101100001000010"]
    for ex in examples :
        print ex, decode( list(ex) )

    import doctest
    verbose=1
    if(verbose):
        doctest.testmod(None,None,None,True)
    else:
        doctest.testmod()
        pass

if __name__ == '__main__': test()