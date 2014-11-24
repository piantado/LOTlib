
import matplotlib.pyplot as plt
from scipy.io import loadmat
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Miscellaneous import logsumexp
import Grammar as G, Hypothesis


#=============================================================================================================
# Generate number set hypotheses
#=============================================================================================================

def normalizing_constant(hypotheses):
    """Estimate normalizing constant Z by logsumexp(posterior scores for all hypotheses)."""
    return logsumexp([h.posterior_score for h in hypotheses])


def make_h0(grammar=G.grammar, **kwargs):
    """Make initial NumberGameHypothesis."""
    return Hypothesis.NumberGameHypothesis(grammar, **kwargs)


#=============================================================================================================
# Infer grammar rule probabilities with human data
#=============================================================================================================

def import_data_from_mat():
    """Script for loading Joshs' number game data.

    Data is originally in probabilitiy (i.e. float) format, so (# yes, # no) pairs are estimated by
    assuming 20 human participants.

    """
    mat = loadmat('../GrammarInference/number_game_data.mat')
    mat_data = mat['data']
    number_game_data = []

    for d in mat_data:

        input_data = d[0][0].tolist()
        output_data = {}

        for i in range(len(d[1][0])):
            key = d[1][0][i]
            associated_prob = d[2][0][i]
            associated_yes = int(associated_prob * 20)
            output_data[key] = (associated_yes, 1-associated_yes)    # est. (# yes, # no) responses

        function_datum = FunctionData(input=input_data, output=output_data)
        number_game_data.append(function_datum)

    return number_game_data


def visualize_probs(probs, dist, rule_name='RULE_'):
    """Visualize results from probs_data_rule."""
    fig, ax = plt.subplots()
    rects = plt.bar(probs, dist)

    plt.xlabel('Grammar Rule Probability')
    plt.ylabel('Pr. of human data')
    plt.title('Prob. of human data given prob. for rule: '+rule_name)
    plt.show()


