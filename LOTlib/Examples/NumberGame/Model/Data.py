"""
Map output number (e.g. 8) to a number of yes/no's.  E.g. (10, 2) ~ (10 yes, 2 no).

"""
from LOTlib import FunctionData
from Utilities import import_data_from_mat


"""
Toy data

"""
grammar_data = [
    # powers of 2:  {n|n = 2^y}
    FunctionData(
        input=[2, 4, 16, 32, 64],
        output={8: (12, 0),
                9: (0, 12),
                10: (0, 12)}
    ),
    # # {n|n = 2^y}  U  {5}
    # FunctionData(
    #     input=[2, 4, 5, 8, 16, 32, 64],
    #     output={8: (12, 0),
    #             9: (1, 11),
    #             10: (6, 6)}
    # ),
    # # {n|n = 2^y}  U  {n|n = 5y}
    # FunctionData(
    #     input=[2, 4, 8, 16, 32, 64,
    #            5, 15, 20, 25, 30, 45, 50, 65, 80, 95],
    #     output={8: (12, 0),
    #             9: (1, 11),
    #             10: (12, 0)}
    # )
]


"""
Import Josh data

"""
mat_data = import_data_from_mat()