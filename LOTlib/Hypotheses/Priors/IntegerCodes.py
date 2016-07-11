"""
This module supplies functions that implement various
codes for integers, some of them self-delimiting.
 (a) regular binary   - encode: dec_to_bin(n,d) ;    decode: bin_to_dec(cl,d,0)
 (b) headless binary  - encode: dec_to_headless(n) ; decode: bin_to_dec(cl,d,1)
 (c) C_alpha(n)       - encode: encoded_alpha(n)   ; decode: get_alpha_integer(cl)
 (d) Fibonacci code (Figital) - encode: to_fiboanacci(n); decoder: from_fibonacci(string) or from_fibonacci2(clist)
 (e) byte code (bases 3, 7, 15, 31,...)
   (This code is allowed to encode integers from 0 upwards, unlike the others, which take n>=1 only)
   
C_alpha(n) is a self-delimiting code for integers.
Neither regular binary nor headless binary is a self-delimiting code,
hence their decoders must be supplied with a parameter d that tells
how many bits to read.

See
   Information Theory, Inference, and Learning Algorithms. (Ch 7: Codes for Integers)
   http://www.inference.phy.cam.ac.uk/mackay/itila/

This package uses the doctest module to test that it is functioning correctly.
IntegerCodes.py is free software (c) David MacKay December 2005. License: GPL
"""
## For license statement see  http://www.gnu.org/copyleft/gpl.html

import sys

def standard_binary( n  ):
    """ n is the number to convert to binary; returns the standard binary representation
    >>> print standard_binary( 17 )
    10001
    >>> print standard_binary( 6 )
    110
    """
    return "1"+dec_to_headless( n )

def dec_to_bin( n , digits ):
    """ n is the number to convert to binary;  digits is the number of bits you want
    Always prints full number of digits
    >>> print dec_to_bin( 17 , 9)
    000010001
    >>> print dec_to_bin( 17 , 5)
    10001
    
    Will behead the standard binary number if requested
    >>> print dec_to_bin( 17 , 4)
    0001
    """
    if(n<0) :
        sys.stderr.write( "warning, negative n not expected\n")
        pass
    i=digits-1
    ans=""
    while i>=0 :
        b = (((1<<i)&n)>0) 
        i -= 1
        ans = ans + str(int(b))
        pass
    return ans
    pass

def ceillog( n ) : ## ceil( log_2 ( n ))   [Used by LZ.py]
    """
    >>> print ceillog(3), ceillog(4), ceillog(5)
    2 2 3
    """
    assert n>=1
    c = 0
    while 2**c<n :
        c += 1
    return c


def to_byte( n, bytesize):
    """
    Self-delimiting code using end of file character.
    Encode integer n>=0 into base B=2**bytesize - 1. Use the all-1 symbol as end-of-file symbol.
    Is this called "Rice Coding"?
    >>> print to_byte( 10, 2 ) ## 10 = 9 + 0 + 1 --> 01 00 01 11
    01000111
    >>> print to_byte( 10, 3 ) ## 10 = 7 + 3     --> 001 011 111
    001011111
    """
    assert(bytesize>1) ## this coder does base 3, 7, 15,...
    assert (n>=0)
    B = (1<<bytesize) - 1
    answer=""
    while n>0 :
        rem = n % B
        answer=dec_to_bin(rem,bytesize)+answer
#        print n,B,rem,answer
        n = n/B
        pass
    answer=answer+"1"*bytesize
    return answer

def from_byte( clist, bytesize):
    """
    Takes a list of binary digits. Returns an integer, and destroys the elements of the
    list that it has read.
    >>> print from_byte( list("01000111"), 2 ) ## 10 = 9 + 0 + 1 --> 01 00 01 11
    10
    >>> print from_byte( list("001011111"), 3 ) ## 10 = 7 + 3     --> 001 011 111
    10
    """
    assert (len(clist)>=0)
    B = (1<<bytesize) - 1
    n=0
    while 1:
        d = bin_to_dec(clist,bytesize)
        if (d == B):
            return n
        else:
            n = n*B + d
            pass
        pass
    pass

def dec_to_headless( n ): 
    """Return the headless binary representation of n, an integer >= 1
    
    >>> [(n,dec_to_headless(n)) for n in range(1,8)]
    [(1, ''), (2, '0'), (3, '1'), (4, '00'), (5, '01'), (6, '10'), (7, '11')]
    >>> dec_to_headless(42)
    '01010'
    """
    assert(n>=1)
    ans=""
    while n>1 :
        b = ((1&n)>0) 
        ans =  str(int(b)) + ans
        n >>= 1
        pass
    return ans
    pass

def bin_to_dec( clist , c , tot=0 ):
    """Implements ordinary binary to integer conversion if tot=0
    and HEADLESS binary to integer if tot=1
    clist is a list of bits; read c of them and turn into an integer.
    The bits that are read from the list are popped from it, i.e., deleted

    Regular binary to decimal 1001 is 9...
    >>> bin_to_dec(  ['1', '0', '0', '1']  ,  4  , 0 )
    9

    Headless binary to decimal [1] 1001 is 25...
    >>> bin_to_dec(  ['1', '0', '0', '1']  ,  4  , 1 )
    25
    """
    while (c>0) :
        assert ( len(clist) > 0 ) ## else we have been fed insufficient bits.
        tot = tot*2 + int(clist.pop(0))
        c-=1
        pass
    return tot

class Fibonacci:
    """
    Returns fibonacci numbers and caches them for efficiency.
    >>> f = Fibonacci(); [f.calculate(i) for i in range(50)]   # doctest:+ELLIPSIS
    [1, 1, 2, 3, 5, 8, 13, 21, ..., 12586269025L]
    """
    def __init__(self):
        self.cache={}
    def calculate(self, n):
        if n < 2: return 1
        try:
            return self.cache[n]
        except:
            return self.cache.setdefault(n, self.calculate(n-2) + self.calculate(n-1))

globalfib = Fibonacci()


def to_fibonacci( r ) :
    """
    Encode the positive integer r using the self-delimiting fibonacci code
    (aka "figital"),
    which has the property that every integer is terminated by "11".
    The decoding of c0c1c2... is c(0)*F(1) + c(1)*F(2) + c(2)*F(3) + ...

    For the Fibonacci code, the implicit distribution is <math>1/n^q</math>, with 
    <math>q = 1/\log_2(\gamma) \simeq 1.44</math>, where <math>\gamma</math> is the golden ratio.
    
    >>> for i in range(1,23): print i,to_fibonacci(i)   # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1  11
    2  011
    3  0011
    4  1011
    5  00011
    6  10011
    7  01011
    8  000011
    9  100011
    10 010011
    11 001011
    12 101011
    ...
    >>> print to_fibonacci(6), from_fibonacci(to_fibonacci(6))
    10011 6
    >>> print to_fibonacci(33), from_fibonacci(to_fibonacci(33))
    10101011 33
    >>> print to_fibonacci(18), from_fibonacci(to_fibonacci(18))
    0001011 18
    """
    i=1
    assert r>=1 ## we encode positive integers only
    ## find thhe biggest fib needed
    while globalfib.calculate(i)<= r:
        i+=1
        pass
    i-=1
#    print i,r,globalfib.calculate(i)
    ans = '1' 
    while i>0:
#        print i,r,globalfib.calculate(i)
        if globalfib.calculate(i)>r:
            ans = '0'+ ans
            pass
        else:
            ans = '1' + ans
            r -= globalfib.calculate(i)
            pass
        i -= 1
        pass
    return ans
    
def from_fibonacci( string ) :
    """
    Decode the positive integer r from self-delimiting fibonacci code
    (aka "figital"),
    which has the property that every integer is terminated by "11".
    The decoding of c0c1c2... is c(0)*F(1) + c(1)*F(2) + c(2)*F(3) + ...
    except we ignore the final 1.
    >>> print from_fibonacci('10101011')
    33
    >>> print from_fibonacci('0001011')
    18
    >>> print from_fibonacci('10011')
    6
    """
    return from_fibonacci2( list(string) )

def from_fibonacci2( slist ):
    global globalfib
    counter = 1 ; ans = 0 ; seenOne = 0 ;
    while 1:
        assert len(slist)>0
        s = slist.pop(0)
        if s == '1':
            if seenOne:
                return ans
            ans += globalfib.calculate(counter)
            seenOne = 1 
            counter+=1
            pass
        elif s == '0':
            counter +=1
            seenOne = 0
            pass
        else:
            import sys
            print >> sys.stderr, "warning, illegal character `s` in from_fiboanacci"
            pass
        pass
    pass
    
def   encoded_omega ( r ) :
    """
    Encode the positive integer r using C_omega
    See
       Information Theory, Inference, and Learning Algorithms. (Ch 7: Codes for Integers)
       http://www.inference.phy.cam.ac.uk/mackay/itila/ 
    >>> print encoded_omega(31)
    10100111110
    >>> print encoded_omega(1)
    0
    >>> print encoded_omega(2)
    100
    >>> print encoded_omega(3)
    110
    """
    return omega_recursion(r)+"0"

def omega_recursion(r):
    ## find the standard binary code for r
    cb = standard_binary(r)
    l = len(cb)-1 ## assert l is floor(log_2(n))
    if l>=1:
        return omega_recursion(l) + cb
    else:
        return ""
    pass

def get_omega_integer( clist ):
    """
    Destructively read a self-delimiting integer from the head of clist and return
    its value.
    The answer "0" is returned if the clist is empty.
    An assertion error message will result if the clist ends before we have finished
    reading.
    >>> for i in [1,3,6,45]: b=encoded_omega(i); cl=list(b); j=get_omega_integer(cl); print i,b,j; assert i==j
    1 0 1
    3 110 3
    6 101100 6
    45 101011011010 45
    """
    if (len(clist)>0 ) :
        return get_omega_recurse( clist, 1 )
    else:
        return 0 ## this is an error state
    pass

def get_omega_recurse(clist, length):
    assert ( len(clist) > 0 ) ## otherwise we have an invalid encoded string
    if ( clist.pop(0) == '0' ) :
        return length
    else :    
        n = bin_to_dec( clist, length, 1 ) ## headless integer
        return get_omega_recurse(clist, n)
        pass
    pass
    

def   encoded_alpha ( r ) :
    """
    Encode the positive integer r using C_alpha
    See
       Information Theory, Inference, and Learning Algorithms. (Ch 7: Codes for Integers)
       http://www.inference.phy.cam.ac.uk/mackay/itila/ 
    >>> print encoded_alpha( 17)
    000010001
    >>> print encoded_alpha( 9)
    0001001
    >>> print encoded_alpha( 63)
    00000111111
    """
    c=0 ; rc =r ; ans=""
    while 1:
        r = (r >> 1)
        if r<1 : break
        ans = ans+"0"
        c += 1
        pass
    ans = ans + dec_to_bin( rc , c+1 )   ## prints the standard binary representation of the number r
    return ans
    pass

def get_alpha_integer ( clist ) :
    """
    Destructively read a self-delimiting integer from the head of clist and return
    its value.
    The answer "0" is returned if the clist is empty.
    An assertion error message will result if the clist ends before we have finished
    reading.
    >>> for i in [1,3,6,42]: b=encoded_alpha(i); cl=list(b); print b,get_alpha_integer(cl)
    1 1
    011 3
    00110 6
    00000101010 42
    """
    if (len(clist)>0 ) :
        c = 0 ## counts number of bits to read
        while ( clist.pop(0) == '0' ) :
            c += 1
            assert ( len(clist) > 0 )
            pass
        #        print "now going to read",c,"bits "
        # ok, we have just read a 1, that was the start of the integer
        r = bin_to_dec( clist , c , 1 )
        return r        
        pass
    else:
        return 0
    pass

def assertions():
    print "Testing byte"
    for j in range(2,7):
        for i in range(1299):
            assert from_byte(list(to_byte(i,j)),j) == i
    print "Testing fibo"
    for i in range(1,12999):
        assert from_fibonacci2(list(to_fibonacci(i))) == i
    print "Testing omega"
    for i in range(1,12999):
        assert get_omega_integer(list(encoded_omega(i))) == i
    print "Testing alpha"
    for i in range(1,12999):
        assert get_alpha_integer(list(encoded_alpha(i))) == i
    pass

def test():
    import doctest
    for i in range(1,24): print "%-3d%s" % ( i,to_fibonacci(i) )
    verbose=1
    if(verbose):
        doctest.testmod(None,None,None,True)
    else:
        doctest.testmod()
        pass
    assertions()
    pass

def oldtest():
    print "testing dec_to_bin"
    testlist = ["print dec_to_bin( 17 , 9)",\
                "print dec_to_bin( 17 , 6)",\
                "print dec_to_bin( 17 , 5)",\
                "print dec_to_bin( 17 , 4)"] 
    for t in testlist :
        print t
        exec t
        pass

    print "\ntesting bin_to_dec"
    binlist = ["1001", "101010" ]
    for b in binlist :
        clist = list( b )
        length = len(clist)
        print "\nRegular binary to decimal", b
        print "bin_to_dec( ",clist," , ",length," , 0 )"
        a = bin_to_dec( clist , length , 0 )
        print a

        print "\nHeadless binary to decimal [1]", b
        clist = list( b )
        print "bin_to_dec( ",clist," , ",length," , 1 )"
        a = bin_to_dec( clist , length , 1 )
        print a
        pass
    pass
        
    print "\ntesting alpha encoder"
    testlist = ["print encoded_alpha( 17)",\
                "print encoded_alpha( 9)",
                "print encoded_alpha( 63)"]
    for t in testlist :
        print t
        exec t
        pass

    print "\ntesting alpha decoder"
    for i in range (1,23):
        print "encoding", i
        b = encoded_alpha( i )
        print " ->",b
        clist = list(b)
        a = get_alpha_integer( clist )
        print " ->",a
        if i!=a :
            print "ERROR"
            pass
        pass
    pass


## from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/111286 
BASE2 = "01"
BASE10 = "0123456789"
BASE16 = "0123456789ABCDEF"
BASE62 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"

def baseconvert(number,fromdigits,todigits):
    """ converts a "number" between two bases of arbitrary digits

    The input number is assumed to be a string of digits from the
    fromdigits string (which is in order of smallest to largest
    digit). The return value is a string of elements from todigits
    (ordered in the same way). The input and output bases are
    determined from the lengths of the digit strings. Negative 
    signs are passed through.

    decimal to binary
    >>> baseconvert(555,BASE10,BASE2)
    '1000101011'

    binary to decimal
    >>> baseconvert('1000101011',BASE2,BASE10)
    '555'

    integer interpreted as binary and converted to decimal (!)
    >>> baseconvert(1000101011,BASE2,BASE10)
    '555'

    base10 to base4
    >>> baseconvert(99,BASE10,"0123")
    '1203'

    base4 to base5 (with alphabetic digits)
    >>> baseconvert(1203,"0123","abcde")
    'dee'

    base5, alpha digits back to base 10
    >>> baseconvert('dee',"abcde",BASE10)
    '99'

    decimal to a base that uses A-Z0-9a-z for its digits
    >>> baseconvert(257938572394L,BASE10,BASE62)
    'E78Lxik'

    ..convert back
    >>> baseconvert('E78Lxik',BASE62,BASE10)
    '257938572394'

    binary to a base with words for digits (the function cannot convert this back)
    >>> baseconvert('1101',BASE2,('Zero','One'))
    'OneOneZeroOne'

    """

    if str(number)[0]=='-':
        number = str(number)[1:]
        neg=1
    else:
        neg=0

    # make an integer out of the number
    x=long(0)
    for digit in str(number):
       x = x*len(fromdigits) + fromdigits.index(digit)
    
    # create the result in base 'len(todigits)'
    res=""
    while x>0:
        digit = x % len(todigits)
        res = todigits[digit] + res
        x /= len(todigits)
    if neg:
        res = "-"+res

    return res

"""
 Note about conversion between bases. If the number is in string form, 
 the constructor for int is designed to handle any base from 2 
 through 36:
    >>> int('FF', 16) # hexadecimal (base 16)
    255
    >>> int('777', 8) # base 8
    511
    >>> int('10000011', 2) # binary
    131
    >>> int('1234', 10) # decimal
    1234
    >>> int('1234') # defaults to decimal if not specified
    1234
"""

## from  http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/65212
def base10toN(num,n):
    """Change a  to a base-n number.
    Up to base-36 is supported without special notation."""
    num_rep={10:'a',
         11:'b',
         12:'c',
         13:'d',
         14:'e',
         15:'f',
         16:'g',
         17:'h',
         18:'i',
         19:'j',
         20:'k',
         21:'l',
         22:'m',
         23:'n',
         24:'o',
         25:'p',
         26:'q',
         27:'r',
         28:'s',
         29:'t',
         30:'u',
         31:'v',
         32:'w',
         33:'x',
         34:'y',
         35:'z'}
    new_num_string=''
    current=num
    while current!=0:
        remainder=current%n
        if 36>remainder>9:
            remainder_string=num_rep[remainder]
        elif remainder>=36:
            remainder_string='('+str(remainder)+')'
        else:
            remainder_string=str(remainder)
        new_num_string=remainder_string+new_num_string
        current=current/n
    return new_num_string

## from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/410672
NOTATION10 = '0123456789'
NOTATION70 = "!'()*-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz~"

class BaseConvert:
    def __init__(self):
        self.notation = NOTATION10

    def _convert(self, n=1, b=10):
        '''
        Private function for doing conversions; returns a list
        '''
        if True not in [ isinstance(n, x) for x in [long, int, float] ]:
            raise TypeError, 'parameters must be numbers'
        converted = []
        quotient, remainder = divmod(n, b)
        converted.append(remainder)
        if quotient != 0:
            converted.extend(self._convert(quotient, b))
        return converted

    def convert(self, n, b):
        '''
        General conversion function
        '''
        nums = self._convert(n, b)
        nums.reverse()
        return self.getNotation(nums)

    def getNotation(self, list_of_remainders):
        '''
        Get the notational representation of the converted number
        '''
        return ''.join([ self.notation[x] for x in list_of_remainders ])

class Base70(BaseConvert):
    '''
    >>> base = Base70()
    >>> base.convert(10)
    '4'
    >>> base.convert(510)
    '1E'
    >>> base.convert(1000)
    '8E'
    >>> base.convert(10000000)
    'N4o4'
    '''
    def __init__(self):
        self.notation = NOTATION70

    def convert(self, n):
        "Convert base 10 to base 70"
        return BaseConvert.convert(self, n, 70)

if __name__ == '__main__': test()