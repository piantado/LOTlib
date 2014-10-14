__author__ = 'eric'

# probability of generating output | grammar, data
def likelihoodOfItemGivenModel(grammar, data, output):
    hypotheses = []  ### [generate a bunch of hypotheses]
    likelihood = 0
    for h in hypotheses:
        likelihood += h.compute_likelihood(data)


# maps output number (e.g. 8) to a number of yes/no's (e.g. [10/2] )
human_in_data = [2, 4, 6]
human_out_data = {
    8: (10, 2),
    12: (5, 7),
    14: (8, 4)
}



def howLikelyIsHumanDataGivenGrammar(grammar, input_data=[], output_data={}):
    return 0

