from LOTlib.DataAndObjects import FunctionData

# Make up some data

# here just doubling x :-> cons(x,x)
data = [
    FunctionData(
        input=[[]],
        output=[[], []]
    ),
    FunctionData(
        input=[[[]]],
        output=[[[]], [[]]]
    ),
]

 # A little more interesting. Squaring: N parens go to N^2
#data = [
        #FunctionData( input=[[  [ [] ] * i  ]], output=[ [] ] * (i**2) ) \
        #for i in xrange(1,10)
       #]
