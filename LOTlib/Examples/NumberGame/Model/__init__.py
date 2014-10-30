
from Data import *          # 'import *' is ok because Data has no imports
from Grammar import grammar
from Hypothesis import NumberSetHypothesis, GrammarProbHypothesis
from Utilities import normalizing_constant, make_h0, random_sample, mh_sample, prob_data_rule, \
    probs_data_rule, prob_data, likelihood_data, get_rule, visualize_probs
import Data, Grammar, Hypothesis, Utilities
