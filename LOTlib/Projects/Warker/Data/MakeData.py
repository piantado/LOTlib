from itertools import product

def unique_chars_set(s):
    s = ''.join(s.split())
    return len(s) == len(set(s))


all_words = [''.join(i) for i in product(['f ','s ','n ','m ','g ','h ','s ','k ','N ','e '], repeat = 3)]

words_with_vowel = []
stim_words = []
possible = []
for w in all_words:
    if 'e' in w:
        s = w.strip(' \t\n\r')
        words_with_vowel.append(s)
        if (w[4]!= 'h') and (w[0]!="N") and w[0]!='e'  and w[4]!= 'e' and w[0]!='s' and w[4]!='f':
            if w[2]=='e':
                possible.append(w.strip(' \t\n\r'))
                if unique_chars_set(w):

                        stim_words.append(w)

stimdict = dict((w.strip(' \t\n\r'), 100) for w in stim_words)

#print stimdict
#print possible





KGall_words = [''.join(i) for i in product(['f ','s ','n ','m ','g ','h ','s ','k ','N ','a ','i '], repeat = 3)]

kaggik_words = []
kaggikvowel=[]
for w in KGall_words:
    w = w.strip(' \t\n\r')
    if 'a' in w or 'i' in w:
        kaggikvowel.append(w)
    if unique_chars_set(w) and (w[4]!= 'h') and (w[0]!="N") and (w[0]!='a' and w[0]!= 'i') and (w[4]!= 'a' and w[4]!= 'i'):
        if w[2]=='a' and w[4]!='k':
            kaggik_words.append(w)
        if w[2]=='i' and w[4]!='g':
            kaggik_words.append(w)

print kaggik_words

KGstimdict = dict((w.strip(' \t\n\r'), 100) for w in kaggik_words)


englishALL = [''.join(i) for i in product(['e ','I ','a ','A ','u ','O ','o ','U ','t ','r ','l ','s ','d ','n ','k ','m ','z ','v ','p ','w ','b ','f ','y ','g ','h ','S ','N ','j ','T '], repeat = 3)]
vowels = ['e','I','a','A','u','U','O','o']
english_vowel=[]
for w in englishALL:
    w = w.strip(' \t\n\r')
    if w[2] in vowels and w[0] not in vowels and w[4] not in vowels:
        english_vowel.append(w)
#print english_vowel

classesALL = [''.join(i) for i in product(['p ','b ','m ','w ','i ','e ','f ','v ','t ','s ','z ','h ','d ','s ','z ','n ','l ','r ','h ','j ','k ','g ','N '], repeat = 3)]
cvowels=['e','i']
classes_vowel=[]
for w in classesALL:
    w = w.strip(' \t\n\r')
    if w[2] in cvowels and w[0] not in cvowels and w[4] not in cvowels:
        classes_vowel.append(w)
#print classes_vowel

