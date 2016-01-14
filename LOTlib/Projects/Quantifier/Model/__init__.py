
from Data import all_objects, TESTING_SET_SIZE, TESTING_SET, target, sample_context, generate_data
from Grammar import grammar
from Hypothesis import MyContext, GriceanQuantifierLexicon
from Utilities import make_my_hypothesis, my_weight_function, gricean_weight, show_baseline_distribution, \
    is_conservative, extract_literal, extract_presup, collapse_undefs, check_counts
import Data, Grammar, Hypothesis, Utilities
