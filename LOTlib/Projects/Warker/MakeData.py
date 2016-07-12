import random

def dellData(bool):
    unrestricted = ['k', 'g', 'm', 'n']
    onsets = ['f', 'h']
    codas = ['s', 'N']
    vowels = ['e']

    onsets += (unrestricted)
    codas += (unrestricted)

    stim = ''
    if(bool):
        stim+=random.choice(onsets)+' '+random.choice(vowels)+' '+random.choice(codas)
    else:
        stim+=random.choice(codas)+' '+random.choice(vowels)+' '+random.choice(onsets)
    return stim


def kaggikData(bool):

    unrestricted = ['f', 's', 'm', 'n']
    onsetsa = ['g', 'h']
    codasa = ['k', 'N']
    onsetsi = ['k','h']
    codasi = ['g','N']
    vowels = ['a','i']

    onsetsa += (unrestricted)
    codasa += (unrestricted)
    onsetsi += (unrestricted)
    codasi+= (unrestricted)

    stim = ''
    v = random.choice(vowels)

    if v == 'a':
        if bool:
            stim+=random.choice(onsetsa)+' '+v+' '+random.choice(codasa)
        else:
            stim+=random.choice(codasa)+' '+v+' '+random.choice(onsetsa)
    else:
        if bool:
            stim+=random.choice(onsetsi)+' '+v+' '+random.choice(codasi)
        else:
            stim+=random.choice(codasi)+' '+v+' '+random.choice(onsetsi)

    return stim








def lotsa(n, size, fn, bool):
    mydata = {}
    setty = set()
    for i in range(0,n):
        mydata.update({fn(bool):size})
        setty.add(fn(bool))
    print(setty)
    return mydata
#lotsa(100, 100, dellData,False)
d = lotsa(200,100, kaggikData,False)
listy = []
for k,v in d.items():
    if 'N' in k or 'k' in k or 'g' in k or 'h' in k:
        listy.append(k)
print (listy)
from LOTlib.Primitives import *
from LOTlib.Miscellaneous import flatten2str
for i in range(50):
    print(flatten2str(if_(flip_(),cons_(cons_(sample_("fsnmgh"),'a'),sample_("fsnmkN")),cons_(cons_(sample_( "fsnmkh"),'i'),sample_("fsnmgN")))))
