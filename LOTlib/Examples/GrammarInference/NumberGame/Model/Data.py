"""
Map output number (e.g. 8) to a number of yes/no's.  E.g. (10, 2) ~ (10 yes, 2 no).

"""
from LOTlib import FunctionData
from Utilities import import_data_from_mat


"""
Toy data

"""
toy_exp_1 = [
    # powers of 2:  {2^y}
    FunctionData(
        input=[2, 4, 16, 32, 64],
        output={8: (12, 0),
                9: (0, 12),
                10: (0, 12)}
    ),
    # {2^y}  U  {5}
    FunctionData(
        input=[2, 4, 5, 8, 16, 32, 64],
        output={8: (12, 0),
                9: (1, 11),
                10: (6, 6)}
    ),
    # {2^y}  U  {5y}
    FunctionData(
        input=[2, 4, 8, 16, 32, 64,
               5, 15, 20, 25, 30, 45, 50, 65, 80, 95],
        output={8: (12, 0),
                9: (1, 11),
                10: (12, 0)}
    )
]

toy_exp_2 = [
    # {2^y + 1}
    FunctionData(
        input=[1, 3, 7, 15, 31, 63],
        output={7: (12, 0),
                15: (12, 0),
                31: (12, 0),
                63: (12, 0),
                8: (0, 12),
                32: (0, 12),
                50: (0, 12)}
    )
]

toy_exp_3 = [
    # {2^y}
    FunctionData(
        input=[2, 4, 8, 16, 32, 64],
        output={8: (12, 0),
                16: (12, 0),
                32: (12, 0),
                64: (12, 0),
                7: (0, 12),
                31: (0, 12),
                53: (0, 12)}
    )
]

toy_single = [
    # {no rule, just 1 datum}   -->  human data emphasizes 2^n, with a little 2*n
    FunctionData(
        input=[2],
        output={7:  (2, 10),
                8:  (12, 0),
                16: (12, 0),
                20: (12, 0),
                30: (4,  8),
                32: (12, 0),
                50: (6,  6),
                64: (12, 0),
                63: (0, 12)}
    )
]


"""
Import Josh data

"""
josh_data = import_data_from_mat()