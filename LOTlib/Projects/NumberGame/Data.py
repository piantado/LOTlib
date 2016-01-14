
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the human data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# We will map tuples of concept-list, set, response to counts.
import pandas
from collections import Counter

def load_human_data(path='numbergame_data.csv'):

    human_data = pandas.read_csv(path, sep=',', low_memory=False, index_col=False)

    human_nyes, human_ntrials = Counter(), Counter()
    for r in xrange(human_data.shape[0]): # for each row
        observed_set = eval(human_data['concept'][r])
        if isinstance(observed_set,int): # ugh fix the formatting
            observed_set = [observed_set]
        observed_set = tuple(observed_set)

        ct = tuple([ observed_set, human_data['target'][r] ])
        rsp = human_data['rating'][r]

        human_ntrials[ct] += 1
        if rsp == 1:
            human_nyes[ct] += 1
    return human_nyes, human_ntrials
