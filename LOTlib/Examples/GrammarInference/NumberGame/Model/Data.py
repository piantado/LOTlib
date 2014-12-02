"""
Map output number (e.g. 8) to a number of yes/no's.  E.g. (10, 2) ~ (10 yes, 2 no).

"""
from LOTlib import FunctionData
from Utilities import import_data_from_mat


"""
Toy data

"""
toy_exp_1 = [
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

toy_exp_2 = [
    FunctionData(
        input=[1, 1, 1, 1, 3, 3, 3, 3, 7, 7, 7, 7, 15, 15, 15, 15, 31, 31, 31, 31, 63, 63, 63, 63],
        output={7: (12, 0),
                15: (12, 0),
                31: (12, 0),
                63: (12, 0),
                8: (0, 12),
                32: (0, 12),
                50: (0, 12)}
    )
]

"""
Import Josh data

"""
josh_data = import_data_from_mat()