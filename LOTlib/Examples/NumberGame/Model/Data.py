from LOTlib import FunctionData

# Map output number (e.g. 8) to a number of yes/no's.  E.g. (10, 2) ~ (10 yes, 2 no).

in1 = [2, 4, 6]
out1 = {
    8: (10, 2),
    12: (5, 7),
    14: (8, 4)
}
data1 = FunctionData(in1, out1)

in2 = [3, 5, 7]
out2 = {
    8: (10, 2),
    12: (5, 7),
    14: (8, 4)
}
data2 = FunctionData(in2, out2)

data = [data1, data2]
