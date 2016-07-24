from itertools import product

def unique_chars_set(s):
    s = ''.join(s.split())
    return len(s) == len(set(s))


all_words = [''.join(i) for i in product(['f ','s ','n ','m ','g ','h ','s ','k ','N ','e '], repeat = 3)]
#print all_words

words_with_vowel = []
stim_words = []

for w in all_words:
    if 'e' in w:
        s = w.strip(' \t\n\r')
        words_with_vowel.append(s)
        if unique_chars_set(w):
            if w[2]=='e':
                stim_words.append(w)
#print words_with_vowel
#print stim_words



KGall_words = [''.join(i) for i in product(['f ','s ','n ','m ','g ','h ','s ','k ','N ','a ','i '], repeat = 3)]
kaggik_words = []
for w in KGall_words:
    w = w.strip(' \t\n\r')
    if unique_chars_set(w) and (w[4]!= 'h') and (w[0]!="N") and (w[0]!='a' and w[0]!= 'i') and (w[4]!= 'a' and w[4]!= 'i'):
        if w[2]=='a' and w[4]!='k':
            kaggik_words.append(w)
        if w[2]=='i' and w[4]!='g':
            kaggik_words.append(w)

print kaggik_words
