
import os
from scipy.io import loadmat
from LOTlib.DataAndObjects import FunctionData


def import_josh_data():
    """Script for loading Joshs' number game data.

    Data is originally in probabilitiy (i.e. float) format, so (# yes, # no) pairs are estimated by
    assuming 20 human participants.

    """
    path = os.getcwd()
    mat = loadmat(path+'/number_game_data.mat')
    mat_data = mat['data']
    number_game_data = []

    for d in mat_data:

        input_data = d[0][0].tolist()
        output_data = {}

        for i in range(len(d[1][0])):
            key = d[1][0][i]
            associated_prob = d[2][0][i]
            associated_yes = int(associated_prob * 20)
            output_data[key] = (associated_yes, 20-associated_yes)    # est. (# yes, # no) responses

        function_datum = FunctionData(input=input_data, output=output_data)
        number_game_data.append(function_datum)

    return number_game_data


josh_data = import_josh_data()