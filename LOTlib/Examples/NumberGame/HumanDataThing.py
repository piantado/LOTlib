__author__ = 'eric'

# probability of generating output | grammar, data
def likelihoodOfItemGivenModel(grammar, data, output):
    [generate a bunch of hypotheses]
    likelihood = 0
    for h in hypotheses:
        likelihood += h.compute_likelihood(data)




def howLikelyIsHumanDataGivenModel(grammar, datas):
