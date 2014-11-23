from LOTlib import FunctionData
from Utilities import import_data_from_mat


# Map output number (e.g. 8) to a number of yes/no's.  E.g. (10, 2) ~ (10 yes, 2 no).
data = [
    FunctionData(
        input=[2, 4, 6],
        output={8: (12, 0),
                9: (4, 8),
                10: (11, 1)}
    ),
    FunctionData(
        input=[3, 5, 7],
        output={8: (0, 12),
                9: (11, 1),
                10: (2, 10)}
    )
]

# Note: this data is in probabilities, not #yes / #no !
mat_data = import_data_from_mat()