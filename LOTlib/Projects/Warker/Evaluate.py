from __future__ import division
import pickle
from scipy.misc import logsumexp
import numpy as np
from LOTlib.Miscellaneous import Infinity, nicelog
from Model import MyHypothesis


print "Loading the hypothesis space . . ."
#load the hypothesis space
spaceset = pickle.load(open("tophyp.pkl", "r"))
space = list(spaceset)



for h in space:
    if(h.value.count_nodes() >= 50 or 'll_counts' not in dir(h)):
       space.remove(h)


posteriors=[]
for h in space:
    posteriors.append(h.posterior_score)
    print h


pdata = logsumexp(posteriors)



'''pbip = -Infinity
pbip = np.exp(logsumexp([nicelog(h.ll_counts['b i p']) - nicelog(sum(h.ll_counts.values())) + (h.posterior_score - pdata) for h in space]))
print "Given 'bim' and 'bop', the probability that I will expect to see 'bip' is:  " + str(pbip)'''

## Never gonna see these?
'''illegals = ['n i k', 'n i h', 'g i f', 'g i k', 'f a h', 'g i m', 'g i s', 'f i k', 'k a n', 's i k', 'k a m', 'N a s', 'N i s', 'g i n', 'N a g', 'N a f', 'N a h', 'N a n', 'k a h', 'N a m', 'n a h', 'k a s', 'f a g', 'k a g', 'n a g', 's a h', 'f i h', 'N i k', 'N i h', 'N i n', 'N i m', 'N i f', 'm i h', 'm i k', 'm a g', 'm a h']

illegal_probs = dict((w, -Infinity) for w in illegals)

#over all the illegals, what are each of their probabilities given the whole space?
for w in illegal_probs:
    illegal_probs[w] = np.exp(logsumexp([nicelog(h.ll_counts[w] + 1e-20) - nicelog(sum(h.ll_counts.values())+(1e-20*len(h.ll_counts.keys()))) + (h.posterior_score - pdata) for h in space]))
print illegal_probs'''

all =['fff', 'ffs', 'ffn', 'ffm', 'ffg', 'ffh', 'ffs', 'ffk', 'ffN', 'ffe', 'fsf', 'fss', 'fsn', 'fsm', 'fsg', 'fsh', 'fss', 'fsk', 'fsN', 'fse', 'fnf', 'fns', 'fnn', 'fnm', 'fng', 'fnh', 'fns', 'fnk', 'fnN', 'fne', 'fmf', 'fms', 'fmn', 'fmm', 'fmg', 'fmh', 'fms', 'fmk', 'fmN', 'fme', 'fgf', 'fgs', 'fgn', 'fgm', 'fgg', 'fgh', 'fgs', 'fgk', 'fgN', 'fge', 'fhf', 'fhs', 'fhn', 'fhm', 'fhg', 'fhh', 'fhs', 'fhk', 'fhN', 'fhe', 'fsf', 'fss', 'fsn', 'fsm', 'fsg', 'fsh', 'fss', 'fsk', 'fsN', 'fse', 'fkf', 'fks', 'fkn', 'fkm', 'fkg', 'fkh', 'fks', 'fkk', 'fkN', 'fke', 'fNf', 'fNs', 'fNn', 'fNm', 'fNg', 'fNh', 'fNs', 'fNk', 'fNN', 'fNe', 'fef', 'fes', 'fen', 'fem', 'feg', 'feh', 'fes', 'fek', 'feN', 'fee', 'sff', 'sfs', 'sfn', 'sfm', 'sfg', 'sfh', 'sfs', 'sfk', 'sfN', 'sfe', 'ssf', 'sss', 'ssn', 'ssm', 'ssg', 'ssh', 'sss', 'ssk', 'ssN', 'sse', 'snf', 'sns', 'snn', 'snm', 'sng', 'snh', 'sns', 'snk', 'snN', 'sne', 'smf', 'sms', 'smn', 'smm', 'smg', 'smh', 'sms', 'smk', 'smN', 'sme', 'sgf', 'sgs', 'sgn', 'sgm', 'sgg', 'sgh', 'sgs', 'sgk', 'sgN', 'sge', 'shf', 'shs', 'shn', 'shm', 'shg', 'shh', 'shs', 'shk', 'shN', 'she', 'ssf', 'sss', 'ssn', 'ssm', 'ssg', 'ssh', 'sss', 'ssk', 'ssN', 'sse', 'skf', 'sks', 'skn', 'skm', 'skg', 'skh', 'sks', 'skk', 'skN', 'ske', 'sNf', 'sNs', 'sNn', 'sNm', 'sNg', 'sNh', 'sNs', 'sNk', 'sNN', 'sNe', 'sef', 'ses', 'sen', 'sem', 'seg', 'seh', 'ses', 'sek', 'seN', 'see', 'nff', 'nfs', 'nfn', 'nfm', 'nfg', 'nfh', 'nfs', 'nfk', 'nfN', 'nfe', 'nsf', 'nss', 'nsn', 'nsm', 'nsg', 'nsh', 'nss', 'nsk', 'nsN', 'nse', 'nnf', 'nns', 'nnn', 'nnm', 'nng', 'nnh', 'nns', 'nnk', 'nnN', 'nne', 'nmf', 'nms', 'nmn', 'nmm', 'nmg', 'nmh', 'nms', 'nmk', 'nmN', 'nme', 'ngf', 'ngs', 'ngn', 'ngm', 'ngg', 'ngh', 'ngs', 'ngk', 'ngN', 'nge', 'nhf', 'nhs', 'nhn', 'nhm', 'nhg', 'nhh', 'nhs', 'nhk', 'nhN', 'nhe', 'nsf', 'nss', 'nsn', 'nsm', 'nsg', 'nsh', 'nss', 'nsk', 'nsN', 'nse', 'nkf', 'nks', 'nkn', 'nkm', 'nkg', 'nkh', 'nks', 'nkk', 'nkN', 'nke', 'nNf', 'nNs', 'nNn', 'nNm', 'nNg', 'nNh', 'nNs', 'nNk', 'nNN', 'nNe', 'nef', 'nes', 'nen', 'nem', 'neg', 'neh', 'nes', 'nek', 'neN', 'nee', 'mff', 'mfs', 'mfn', 'mfm', 'mfg', 'mfh', 'mfs', 'mfk', 'mfN', 'mfe', 'msf', 'mss', 'msn', 'msm', 'msg', 'msh', 'mss', 'msk', 'msN', 'mse', 'mnf', 'mns', 'mnn', 'mnm', 'mng', 'mnh', 'mns', 'mnk', 'mnN', 'mne', 'mmf', 'mms', 'mmn', 'mmm', 'mmg', 'mmh', 'mms', 'mmk', 'mmN', 'mme', 'mgf', 'mgs', 'mgn', 'mgm', 'mgg', 'mgh', 'mgs', 'mgk', 'mgN', 'mge', 'mhf', 'mhs', 'mhn', 'mhm', 'mhg', 'mhh', 'mhs', 'mhk', 'mhN', 'mhe', 'msf', 'mss', 'msn', 'msm', 'msg', 'msh', 'mss', 'msk', 'msN', 'mse', 'mkf', 'mks', 'mkn', 'mkm', 'mkg', 'mkh', 'mks', 'mkk', 'mkN', 'mke', 'mNf', 'mNs', 'mNn', 'mNm', 'mNg', 'mNh', 'mNs', 'mNk', 'mNN', 'mNe', 'mef', 'mes', 'men', 'mem', 'meg', 'meh', 'mes', 'mek', 'meN', 'mee', 'gff', 'gfs', 'gfn', 'gfm', 'gfg', 'gfh', 'gfs', 'gfk', 'gfN', 'gfe', 'gsf', 'gss', 'gsn', 'gsm', 'gsg', 'gsh', 'gss', 'gsk', 'gsN', 'gse', 'gnf', 'gns', 'gnn', 'gnm', 'gng', 'gnh', 'gns', 'gnk', 'gnN', 'gne', 'gmf', 'gms', 'gmn', 'gmm', 'gmg', 'gmh', 'gms', 'gmk', 'gmN', 'gme', 'ggf', 'ggs', 'ggn', 'ggm', 'ggg', 'ggh', 'ggs', 'ggk', 'ggN', 'gge', 'ghf', 'ghs', 'ghn', 'ghm', 'ghg', 'ghh', 'ghs', 'ghk', 'ghN', 'ghe', 'gsf', 'gss', 'gsn', 'gsm', 'gsg', 'gsh', 'gss', 'gsk', 'gsN', 'gse', 'gkf', 'gks', 'gkn', 'gkm', 'gkg', 'gkh', 'gks', 'gkk', 'gkN', 'gke', 'gNf', 'gNs', 'gNn', 'gNm', 'gNg', 'gNh', 'gNs', 'gNk', 'gNN', 'gNe', 'gef', 'ges', 'gen', 'gem', 'geg', 'geh', 'ges', 'gek', 'geN', 'gee', 'hff', 'hfs', 'hfn', 'hfm', 'hfg', 'hfh', 'hfs', 'hfk', 'hfN', 'hfe', 'hsf', 'hss', 'hsn', 'hsm', 'hsg', 'hsh', 'hss', 'hsk', 'hsN', 'hse', 'hnf', 'hns', 'hnn', 'hnm', 'hng', 'hnh', 'hns', 'hnk', 'hnN', 'hne', 'hmf', 'hms', 'hmn', 'hmm', 'hmg', 'hmh', 'hms', 'hmk', 'hmN', 'hme', 'hgf', 'hgs', 'hgn', 'hgm', 'hgg', 'hgh', 'hgs', 'hgk', 'hgN', 'hge', 'hhf', 'hhs', 'hhn', 'hhm', 'hhg', 'hhh', 'hhs', 'hhk', 'hhN', 'hhe', 'hsf', 'hss', 'hsn', 'hsm', 'hsg', 'hsh', 'hss', 'hsk', 'hsN', 'hse', 'hkf', 'hks', 'hkn', 'hkm', 'hkg', 'hkh', 'hks', 'hkk', 'hkN', 'hke', 'hNf', 'hNs', 'hNn', 'hNm', 'hNg', 'hNh', 'hNs', 'hNk', 'hNN', 'hNe', 'hef', 'hes', 'hen', 'hem', 'heg', 'heh', 'hes', 'hek', 'heN', 'hee', 'sff', 'sfs', 'sfn', 'sfm', 'sfg', 'sfh', 'sfs', 'sfk', 'sfN', 'sfe', 'ssf', 'sss', 'ssn', 'ssm', 'ssg', 'ssh', 'sss', 'ssk', 'ssN', 'sse', 'snf', 'sns', 'snn', 'snm', 'sng', 'snh', 'sns', 'snk', 'snN', 'sne', 'smf', 'sms', 'smn', 'smm', 'smg', 'smh', 'sms', 'smk', 'smN', 'sme', 'sgf', 'sgs', 'sgn', 'sgm', 'sgg', 'sgh', 'sgs', 'sgk', 'sgN', 'sge', 'shf', 'shs', 'shn', 'shm', 'shg', 'shh', 'shs', 'shk', 'shN', 'she', 'ssf', 'sss', 'ssn', 'ssm', 'ssg', 'ssh', 'sss', 'ssk', 'ssN', 'sse', 'skf', 'sks', 'skn', 'skm', 'skg', 'skh', 'sks', 'skk', 'skN', 'ske', 'sNf', 'sNs', 'sNn', 'sNm', 'sNg', 'sNh', 'sNs', 'sNk', 'sNN', 'sNe', 'sef', 'ses', 'sen', 'sem', 'seg', 'seh', 'ses', 'sek', 'seN', 'see', 'kff', 'kfs', 'kfn', 'kfm', 'kfg', 'kfh', 'kfs', 'kfk', 'kfN', 'kfe', 'ksf', 'kss', 'ksn', 'ksm', 'ksg', 'ksh', 'kss', 'ksk', 'ksN', 'kse', 'knf', 'kns', 'knn', 'knm', 'kng', 'knh', 'kns', 'knk', 'knN', 'kne', 'kmf', 'kms', 'kmn', 'kmm', 'kmg', 'kmh', 'kms', 'kmk', 'kmN', 'kme', 'kgf', 'kgs', 'kgn', 'kgm', 'kgg', 'kgh', 'kgs', 'kgk', 'kgN', 'kge', 'khf', 'khs', 'khn', 'khm', 'khg', 'khh', 'khs', 'khk', 'khN', 'khe', 'ksf', 'kss', 'ksn', 'ksm', 'ksg', 'ksh', 'kss', 'ksk', 'ksN', 'kse', 'kkf', 'kks', 'kkn', 'kkm', 'kkg', 'kkh', 'kks', 'kkk', 'kkN', 'kke', 'kNf', 'kNs', 'kNn', 'kNm', 'kNg', 'kNh', 'kNs', 'kNk', 'kNN', 'kNe', 'kef', 'kes', 'ken', 'kem', 'keg', 'keh', 'kes', 'kek', 'keN', 'kee', 'Nff', 'Nfs', 'Nfn', 'Nfm', 'Nfg', 'Nfh', 'Nfs', 'Nfk', 'NfN', 'Nfe', 'Nsf', 'Nss', 'Nsn', 'Nsm', 'Nsg', 'Nsh', 'Nss', 'Nsk', 'NsN', 'Nse', 'Nnf', 'Nns', 'Nnn', 'Nnm', 'Nng', 'Nnh', 'Nns', 'Nnk', 'NnN', 'Nne', 'Nmf', 'Nms', 'Nmn', 'Nmm', 'Nmg', 'Nmh', 'Nms', 'Nmk', 'NmN', 'Nme', 'Ngf', 'Ngs', 'Ngn', 'Ngm', 'Ngg', 'Ngh', 'Ngs', 'Ngk', 'NgN', 'Nge', 'Nhf', 'Nhs', 'Nhn', 'Nhm', 'Nhg', 'Nhh', 'Nhs', 'Nhk', 'NhN', 'Nhe', 'Nsf', 'Nss', 'Nsn', 'Nsm', 'Nsg', 'Nsh', 'Nss', 'Nsk', 'NsN', 'Nse', 'Nkf', 'Nks', 'Nkn', 'Nkm', 'Nkg', 'Nkh', 'Nks', 'Nkk', 'NkN', 'Nke', 'NNf', 'NNs', 'NNn', 'NNm', 'NNg', 'NNh', 'NNs', 'NNk', 'NNN', 'NNe', 'Nef', 'Nes', 'Nen', 'Nem', 'Neg', 'Neh', 'Nes', 'Nek', 'NeN', 'Nee', 'eff', 'efs', 'efn', 'efm', 'efg', 'efh', 'efs', 'efk', 'efN', 'efe', 'esf', 'ess', 'esn', 'esm', 'esg', 'esh', 'ess', 'esk', 'esN', 'ese', 'enf', 'ens', 'enn', 'enm', 'eng', 'enh', 'ens', 'enk', 'enN', 'ene', 'emf', 'ems', 'emn', 'emm', 'emg', 'emh', 'ems', 'emk', 'emN', 'eme', 'egf', 'egs', 'egn', 'egm', 'egg', 'egh', 'egs', 'egk', 'egN', 'ege', 'ehf', 'ehs', 'ehn', 'ehm', 'ehg', 'ehh', 'ehs', 'ehk', 'ehN', 'ehe', 'esf', 'ess', 'esn', 'esm', 'esg', 'esh', 'ess', 'esk', 'esN', 'ese', 'ekf', 'eks', 'ekn', 'ekm', 'ekg', 'ekh', 'eks', 'ekk', 'ekN', 'eke', 'eNf', 'eNs', 'eNn', 'eNm', 'eNg', 'eNh', 'eNs', 'eNk', 'eNN', 'eNe', 'eef', 'ees', 'een', 'eem', 'eeg', 'eeh', 'ees', 'eek', 'eeN', 'eee']

all_with_vowel = ['f f e', 'f s e', 'f n e', 'f m e', 'f g e', 'f h e', 'f s e', 'f k e', 'f N e', 'f e f', 'f e s', 'f e n', 'f e m', 'f e g', 'f e h', 'f e s', 'f e k', 'f e N', 'f e e', 's f e', 's s e', 's n e', 's m e', 's g e', 's h e', 's s e', 's k e', 's N e', 's e f', 's e s', 's e n', 's e m', 's e g', 's e h', 's e s', 's e k', 's e N', 's e e', 'n f e', 'n s e', 'n n e', 'n m e', 'n g e', 'n h e', 'n s e', 'n k e', 'n N e', 'n e f', 'n e s', 'n e n', 'n e m', 'n e g', 'n e h', 'n e s', 'n e k', 'n e N', 'n e e', 'm f e', 'm s e', 'm n e', 'm m e', 'm g e', 'm h e', 'm s e', 'm k e', 'm N e', 'm e f', 'm e s', 'm e n', 'm e m', 'm e g', 'm e h', 'm e s', 'm e k', 'm e N', 'm e e', 'g f e', 'g s e', 'g n e', 'g m e', 'g g e', 'g h e', 'g s e', 'g k e', 'g N e', 'g e f', 'g e s', 'g e n', 'g e m', 'g e g', 'g e h', 'g e s', 'g e k', 'g e N', 'g e e', 'h f e', 'h s e', 'h n e', 'h m e', 'h g e', 'h h e', 'h s e', 'h k e', 'h N e', 'h e f', 'h e s', 'h e n', 'h e m', 'h e g', 'h e h', 'h e s', 'h e k', 'h e N', 'h e e', 's f e', 's s e', 's n e', 's m e', 's g e', 's h e', 's s e', 's k e', 's N e', 's e f', 's e s', 's e n', 's e m', 's e g', 's e h', 's e s', 's e k', 's e N', 's e e', 'k f e', 'k s e', 'k n e', 'k m e', 'k g e', 'k h e', 'k s e', 'k k e', 'k N e', 'k e f', 'k e s', 'k e n', 'k e m', 'k e g', 'k e h', 'k e s', 'k e k', 'k e N', 'k e e', 'N f e', 'N s e', 'N n e', 'N m e', 'N g e', 'N h e', 'N s e', 'N k e', 'N N e', 'N e f', 'N e s', 'N e n', 'N e m', 'N e g', 'N e h', 'N e s', 'N e k', 'N e N', 'N e e', 'e f f', 'e f s', 'e f n', 'e f m', 'e f g', 'e f h', 'e f s', 'e f k', 'e f N', 'e f e', 'e s f', 'e s s', 'e s n', 'e s m', 'e s g', 'e s h', 'e s s', 'e s k', 'e s N', 'e s e', 'e n f', 'e n s', 'e n n', 'e n m', 'e n g', 'e n h', 'e n s', 'e n k', 'e n N', 'e n e', 'e m f', 'e m s', 'e m n', 'e m m', 'e m g', 'e m h', 'e m s', 'e m k', 'e m N', 'e m e', 'e g f', 'e g s', 'e g n', 'e g m', 'e g g', 'e g h', 'e g s', 'e g k', 'e g N', 'e g e', 'e h f', 'e h s', 'e h n', 'e h m', 'e h g', 'e h h', 'e h s', 'e h k', 'e h N', 'e h e', 'e s f', 'e s s', 'e s n', 'e s m', 'e s g', 'e s h', 'e s s', 'e s k', 'e s N', 'e s e', 'e k f', 'e k s', 'e k n', 'e k m', 'e k g', 'e k h', 'e k s', 'e k k', 'e k N', 'e k e', 'e N f', 'e N s', 'e N n', 'e N m', 'e N g', 'e N h', 'e N s', 'e N k', 'e N N', 'e N e', 'e e f', 'e e s', 'e e n', 'e e m', 'e e g', 'e e h', 'e e s', 'e e k', 'e e N', 'e e e']

test = dict((w, -Infinity) for w in all_with_vowel)
for w in test:
    test[w] = np.exp(logsumexp([nicelog(h.ll_counts[w] + 1e-6) - nicelog(sum(h.ll_counts.values())+(1e-6*len(h.ll_counts.keys()))) + (h.posterior_score - pdata) for h in space]))
print test.keys()
print test.values()





# what I want to do: calculate percentages of legal errors
# make the space
# "present" a small subset of legal words (d)
# P(H|d) = P(d|H)P(H) <-- how do I update you?
# could use previous posterior as this new version's prior:
# P(H|d1) = P(d1|H)P(H|d0)



with open('probabilities2R', 'w') as f:
    for k,v in test.iteritems():
        f.write(k + ',')
        f.write(str(v))
        f.write('\n')



