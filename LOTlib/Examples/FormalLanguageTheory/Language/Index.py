from LOTlib.Examples.FormalLanguageTheory.Language.An import An
from LOTlib.Examples.FormalLanguageTheory.Language.AnBn import AnBn
from LOTlib.Examples.FormalLanguageTheory.Language.AnB2n import AnB2n
from LOTlib.Examples.FormalLanguageTheory.Language.AnCstarBn import AnCstarBn
from LOTlib.Examples.FormalLanguageTheory.Language.AnBnCn import AnBnCn
from LOTlib.Examples.FormalLanguageTheory.Language.Dyck import Dyck
from LOTlib.Examples.FormalLanguageTheory.Language.SimpleEnglish import SimpleEnglish


def instance(s):
    try:
        return {
            'An': An(),
            'AnBn': AnBn(),
            'AnB2n': AnB2n(),
            'Dyck': Dyck(),
            'AnCstarBn': AnCstarBn(),
            'AnBnCn': AnBnCn(),
            'SimpleEnglish': SimpleEnglish()
        }[s]
    except:
        raise ValueError