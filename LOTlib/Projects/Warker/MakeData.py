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


def kaggikData():
    onsetsg = ['g','f','s','m','n']
    codask = ['k','f','s','m','n']
    onsetsk = ['k','f','s','m','n']
    codasg = ['g','f','s','m','n']
    vowels = ['a', 'i']

    stim = ''
    v = random.choice(vowels)
    if v == 'a':
        stim+=random.choice(onsetsg)+' '+v+' '+random.choice(codask)
    else:
        stim+=random.choice(onsetsk)+' '+v+' '+random.choice(codasg)
    return stim





def lotsa(n, size, fn, bool):
    mydata = {}
    setty = set()
    for i in range(0,n):
        mydata.update({fn(bool):size})
        setty.add(fn(bool))
    print(setty)
    return mydata

lotsa(100, 100, dellData,False)