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
                63: (0, 12),
                64: (12, 0)}
    )
]


toy_1n = [
    FunctionData(
        input=[2],
        output={2: (9, 3),         # should be high with 2n
                4: (9, 3),
                8: (9, 3),
                3: (9, 3),         # should be high with 3n
                9: (9, 3),
                15: (9, 3),
                6: (9, 3),         # should be high with 2n OR 3n
                12: (9, 3),
                18: (9, 3),
                11: (9, 3),        # should be high with 1n only
                13: (9, 3),
                17: (9, 3)}
    )
]

toy_2n = [
    FunctionData(
        input=[2],
        output={2: (12, 0),         # should be high with 2n
                4: (12, 0),
                8: (12, 0),
                3: (0, 12),         # should be high with 3n
                9: (0, 12),
                15: (0, 12),
                6: (12, 0),         # should be high with 2n OR 3n
                12: (12, 0),
                18: (12, 0),
                11: (0, 12),        # should be high with 1n only
                13: (0, 12),
                17: (0, 12)}
    )
]

toy_3n = [
    FunctionData(
        input=[2],
        output={2: (0, 12),         # should be high with 2n
                4: (0, 12),
                8: (0, 12),
                3: (12, 0),         # should be high with 3n
                9: (12, 0),
                15: (12, 0),
                6: (12, 0),         # should be high with 2n OR 3n
                12: (12, 0),
                18: (12, 0),
                11: (0, 12),        # should be high with 1n only
                13: (0, 12),
                17: (0, 12)}
    )
]


"""
Import Josh data

"""
josh_data = import_data_from_mat()