from An import An
from AnBn import AnBn
from AnB2n import AnB2n
from AnCstarBn import AnCstarBn
from AnBnCn import AnBnCn
from Dyck import Dyck
from SimpleEnglish import SimpleEnglish
from LongDependency import LongDependency
from ABn import ABn


def instance(s, max_length):
    try:
        return ({
            'An': An,
            'AnBn': AnBn,
            'AnB2n': AnB2n,
            'Dyck': Dyck,
            'AnCstarBn': AnCstarBn,
            'AnBnCn': AnBnCn,
            'SimpleEnglish': SimpleEnglish,
            'LongDependency': LongDependency,
            'ABn': ABn
        }[s])(max_length=max_length)
    except:
        raise ValueError