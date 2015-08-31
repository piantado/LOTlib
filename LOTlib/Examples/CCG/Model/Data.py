from LOTlib.DataAndObjects import UtteranceData, Context
from Grammar import OBJECTS
from Utilities import str2sen

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up the data

# this will be referenced in every UtteranceData, and at the end we'll use it to do all possible strings
possible_utterances = []

# These are always true
BASE_FACTS = [("MAN", "JOHN"),("MAN", "BILL"),("WOMAN", "SUSAN"),("WOMAN", "MARY")]


data = [
    UtteranceData(utterance=str2sen('john saw mary'),
                  context=Context(OBJECTS, BASE_FACTS+[("SAW", "JOHN", "MARY")]),
                  possible_utterances=possible_utterances),
    UtteranceData(utterance=str2sen('mary saw john'),
                  context=Context(OBJECTS, BASE_FACTS+[("SAW", "MARY", "JOHN")]),
                  possible_utterances=possible_utterances),

    UtteranceData(utterance=str2sen('mary smiled'),
                  context=Context(OBJECTS, BASE_FACTS+[("SMILED", "MARY")]),
                  possible_utterances=possible_utterances),
    UtteranceData(utterance=str2sen('john smiled'),
                  context=Context(OBJECTS, BASE_FACTS+[("SMILED", "JOHN")]),
                  possible_utterances=possible_utterances),
    UtteranceData(utterance=str2sen('bill smiled'),
                  context=Context(OBJECTS, BASE_FACTS+[("SMILED", "BILL")]),
                  possible_utterances=possible_utterances),
    UtteranceData(utterance=str2sen('susan smiled'),
                  context=Context(OBJECTS, BASE_FACTS+[("SMILED", "SUSAN")]),
                  possible_utterances=possible_utterances),

    UtteranceData(utterance=str2sen('john is man'),
                  context=Context(OBJECTS, BASE_FACTS+[("MAN", "JOHN")]),
                  possible_utterances=possible_utterances),
    UtteranceData(utterance=str2sen('bill is man'),
                  context=Context(OBJECTS, BASE_FACTS+[("MAN", "BILL")]),
                  possible_utterances=possible_utterances),
    UtteranceData(utterance=str2sen('mary is woman'),
                  context=Context(OBJECTS, BASE_FACTS+[("WOMAN", "MARY")]),
                  possible_utterances=possible_utterances),
    UtteranceData(utterance=str2sen('susan is woman'),
                  context=Context(OBJECTS, BASE_FACTS+[("WOMAN", "SUSAN")]),
                  possible_utterances=possible_utterances),


    UtteranceData(utterance=str2sen('every man smiled'),
                  context=Context(OBJECTS, BASE_FACTS+[("SMILED", "JOHN"), ("SMILED", "BILL")]),
                  possible_utterances=possible_utterances),
    UtteranceData(utterance=str2sen('every woman smiled'),
                  context=Context(OBJECTS, BASE_FACTS+[("SMILED", "MARY"), ("SMILED", "SUSAN")]),
                  possible_utterances=possible_utterances),
    UtteranceData(utterance=str2sen('every man laughed'),
                  context=Context(OBJECTS, BASE_FACTS+[("LAUGHED", "JOHN"), ("LAUGHED", "BILL")]),
                  possible_utterances=possible_utterances),
    UtteranceData(utterance=str2sen('every woman laughed'),
                  context=Context(OBJECTS, BASE_FACTS+[("LAUGHED", "MARY"), ("LAUGHED", "SUSAN")]),
                  possible_utterances=possible_utterances)
]

# For simplicity here, just treat each possible utterance as
for di in data:
    possible_utterances.append( di.utterance )

# keep track of all the words
all_words = set()
for di in data:
    for w in di.utterance: all_words.add(w)

def make_data(*args, **kwargs):
    return data