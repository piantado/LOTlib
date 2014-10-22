#		1NP sentences
# NP V
"(S (NP.1 Mary) (VP (V blinked)))", \
# PRON V
"(S (NP.1 She) (VP (V blinked)))", \


#		2NP sentences
# NP VP NP
"(S (NP.1 Mary) (VP (V kicked) (NP.2 Bill)))", \
# NP VP PRON
"(S (NP.1 Bill) (VP (V shaved) (NP.2 him)))", \
# NP VP REFL
"(S (NP.1 Bill) (VP (V shaved) (NP.1 himself)))", \
# PRON VP NP
"(S (NP.1 He) (VP (V killed) (NP.2 Kenny)))", \
# PRON VP PRON
"(S (NP.1 He) (VP (V killed) (NP.2 him)))", \
# PRON VP REFL
"(S (NP.1 He) (VP (V killed) (NP.1 himself)))", \

#		3NP sentences
# NP VP NP NP
"(S (NP.1 Joe) (VP (V owes) (NP.2 Mary) (NP.3 money)))", \
# NP VP PRON NP
"(S (NP.1 Joe) (VP (V owes) (NP.2 her) (NP.3 money)))", \
# NP VP REFL NP
"(S (NP.1 Joe) (VP (V refuses) (NP.1 himself) (NP.3 vacation)))", \
# PRON VP NP NP
"(S (NP.1 He) (VP (V owes) (NP.2 Mary) (NP.3 money)))", \
# PRON VP PRON NP
"(S (NP.1 He) (VP (V owes) (NP.2 her) (NP.3 money)))", \
# PRON VP REFL NP
"(S (NP.1 He) (VP (V refuses) (NP.1 himself) (NP.3 vacation)))", \

#		Possessive Subjects ??When tagging possessives the NP.1 and the NP.3 should be indexed correct?
# NP(NP NP) VP NP
"(S (NP.1 (NP.2 Mario) (POS -s) (N brother)) (VP (V hated) (NP.4 Bowser)))", \
# NP(NP NP) VP PRON
"(S (NP.1 (NP.2 Mario) (POS -s) (N brother)) (VP (V hated) (NP.4 him)))", \
"(S (NP.1 (NP.2 Mario) (POS -s) (N brother)) (VP (V hated) (NP.2 him)))", \
# NP(NP NP) VP REFL
"(S (NP.1 (NP.2 Mario) (POS -s) (N brother)) (VP (V hated) (NP.1 himself)))", \
# NP(PRON NP) VP NP
"(S (NP.1 (NP.2 His) (POS -s) (N brother)) (VP (V hated) (NP.4 Bowser)))", \
# NP(PRON NP) VP PRON
"(S (NP.1 (NP.2 His) (POS -s) (N brother)) (VP (V hated) (NP.2 him)))", \
"(S (NP.1 (NP.2 His) (POS -s) (N brother)) (VP (V hated) (NP.4 him)))", \
# NP(PRON NP) VP REFL
"(S (NP.1 (NP.2 His) (POS -s) (N brother)) (VP (V hated) (NP.1 himself)))", \

#		Subject Subject Relative Clauses
# NP(NP S(VP NP)) VP NP
"(S (NP.1 (NP.1 (DET The) (N assassin) ) (CC that) (S (VP (V hit) (NP.2 Joe)))) (VP (V killed) (NP.3 Max)))", \
# NP(NP S(VP NP)) VP PRON
"(S (NP.1 (NP.1 (DET The) (N assassin) ) (CC that) (S (VP (V hit) (NP.2 Joe)))) (VP (V killed) (NP.2 him)))", \
"(S (NP.1 (NP.1 (DET The) (N assassin) ) (CC that) (S (VP (V hit) (NP.2 Joe)))) (VP (V killed) (NP.4 him)))", \
# NP(NP S(VP NP)) VP REFL
"(S (NP.1 (NP.1 (DET The) (N assassin) ) (CC that) (S (VP (V hit) (NP.2 Joe)))) (VP (V killed) (NP.1 himself)))", \
# NP(NP S(VP PRO)) VP NP
"(S (NP.1 (NP.1 (DET The) (N assassin) ) (CC that) (S (VP (V hit) (NP.2 him)))) (VP (V killed) (NP.2 Joe)))", \
"(S (NP.1 (NP.1 (DET The) (N assassin) ) (CC that) (S (VP (V hit) (NP.2 him)))) (VP (V killed) (NP.3 Joe)))", \
# NP(NP S(VP PRO)) VP PRO
"(S (NP.1 (NP.1 (DET The) (N assassin) ) (CC that) (S (VP (V hit) (NP.2 him)))) (VP (V killed) (NP.2 him)))", \
# NP(NP S(VP PRO)) VP REFL
"(S (NP.1 (NP.1 (DET The) (N assassin) ) (CC that) (S (VP (V hit) (NP.2 him)))) (VP (V killed) (NP.1 himself)))", \
# NP(NP S(VP REFL)) VP NP
"(S (NP.1 (NP.1 (DET The) (N assassin) ) (CC that) (S (VP (V hit) (NP.1 himself)))) (VP (V killed) (NP.2 Joe)))", \
# NP(NP S(VP REFL)) VP PRO
"(S (NP.1 (NP.1 (DET The) (N assassin) ) (CC that) (S (VP (V hit) (NP.1 himself)))) (VP (V killed) (NP.2 him)))", \
# NP(NP S(VP REFL)) VP REFL
"(S (NP.1 (NP.1 (DET The) (N assassin) ) (CC that) (S (VP (V hit) (NP.1 himself)))) (VP (V killed) (NP.1 himself)))", \

#		Subject Object Relative Clauses
# NP(NP S(NP VP)) VP NP
"(S (NP.1 (NP.1 (DET The) (N actress) ) (CC that) (S (NP.2 Mary) (VP V(liked)))) (VP (V hated) (NP.3 Anne)))", \
# NP(NP S(NP VP)) VP PRO
"(S (NP.1 (NP.1 (DET The) (N actress) ) (CC that) (S (NP.2 Mary) (VP V(liked)))) (VP (V hated) (NP.3 her)))", \
"(S (NP.1 (NP.1 (DET The) (N actress) ) (CC that) (S (NP.2 Mary) (VP V(liked)))) (VP (V hated) (NP.2 her)))", \
# NP(NP S(NP VP)) VP REFL
"(S (NP.1 (NP.1 (DET The) (N actress) ) (CC that) (S (NP.2 Mary) (VP V(liked)))) (VP (V hated) (NP.1 herself)))", \
# NP(NP S(PRO VP)) VP NP
"(S (NP.1 (NP.1 (DET The) (N actress) ) (CC that) (S (NP.2 she) (VP V(liked)))) (VP (V hated) (NP.3 Anne)))", \
"(S (NP.1 (NP.1 (DET The) (N actress) ) (CC that) (S (NP.2 she) (VP V(liked)))) (VP (V hated) (NP.2 Anne)))", \
# NP(NP S(PRO VP)) VP PRO
"(S (NP.1 (NP.1 (DET The) (N actress) ) (CC that) (S (NP.2 she) (VP V(liked)))) (VP (V hated) (NP.2 her)))", \
# NP(NP S(PRO VP)) VP REFL
"(S (NP.1 (NP.1 (DET The) (N actress) ) (CC that) (S (NP.2 she) (VP V(liked)))) (VP (V hated) (NP.1 herself)))", \

#		Object Subject Relative Clauses
# NP VP NP(NP S(VP NP))
"(S (NP.1 Martha) (VP (V hated) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (VP V(liked) (NP.3 Maria))))))", \
# NP VP NP(NP S(VP PRO))
"(S (NP.1 Martha) (VP (V hated) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (VP V(liked) (NP.1 her))))))", \
"(S (NP.1 Martha) (VP (V hated) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (VP V(liked) (NP.3 her))))))", \
# NP VP NP(NP S(VP REFL))
"(S (NP.1 Martha) (VP (V hated) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (VP V(liked) (NP.2 herself))))))", \
# PRO VP NP(NP S(VP NP))
"(S (NP.1 She) (VP (V hated) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (VP V(liked) (NP.3 Maria))))))", \
# PRO VP NP(NP S(VP PRO))
"(S (NP.1 She) (VP (V hated) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (VP V(liked) (NP.1 her))))))", \
"(S (NP.1 She) (VP (V hated) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (VP V(liked) (NP.3 her))))))", \
# PRO VP NP(NP S(VP REFL))
"(S (NP.1 She) (VP (V hated) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (VP V(liked) (NP.2 herself))))))", \

#		Object Object Relative Clauses
# NP VP NP(NP S(NP VP))
"(S (NP.1 Jeff) (VP (V knew) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (NP.3 Steve) (VP V(liked))))))", \
# NP VP NP(NP S(PRO VP))
"(S (NP.1 Jeff) (VP (V knew) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (NP.1 he) (VP V(liked))))))", \
"(S (NP.1 Jeff) (VP (V knew) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (NP.3 he) (VP V(liked))))))", \
# PRO VP NP(NP S(NP VP))
"(S (NP.1 He) (VP (V knew) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (NP.3 Steve) (VP V(liked))))))", \
# PRO VP NP(NP S(PRO VP))
"(S (NP.1 He) (VP (V knew) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (NP.1 he) (VP V(liked))))))", \
"(S (NP.1 He) (VP (V knew) (NP.2 (NP.2 (DET the) (N actress) ) (CC that) (S (NP.3 he) (VP V(liked))))))", \

#		Complement Clauses Subject
#	Intransitive
# NP VP S( NP VP)
"(S (NP.1 Donald) (VP (V knows) (S (NP.2 Minnie) (VP (V smells)))))",\
# NP VP S( PRO VP)
"(S (NP.1 Donald) (VP (V knows) (S (NP.1 he) (VP (V smells)))))",\
"(S (NP.1 Donald) (VP (V knows) (S (NP.2 he) (VP (V smells)))))",\
# PRO VP S( NP VP)
"(S (NP.1 He) (VP (V knows) (S (NP.2 Donald) (VP (V smells)))))",\
# PRO VP S( PRO VP)
"(S (NP.1 He) (VP (V knows) (S (NP.1 he) (VP (V smells)))))",\
"(S (NP.1 He) (VP (V knows) (S (NP.2 he) (VP (V smells)))))",\
#	Transitive
# NP VP S( NP VP NP)
"(S (NP.1 Mickey) (VP (V knows) (S (NP.2 Donald) (VP (V owes) (NP.3 Jack)))))",\
# NP VP S( NP VP PRO)
"(S (NP.1 Mickey) (VP (V knows) (S (NP.2 Donald) (VP (V owes) (NP.3 him)))))",\
"(S (NP.1 Mickey) (VP (V knows) (S (NP.2 Donald) (VP (V owes) (NP.1 him)))))",\
# NP VP S( NP VP REFL)
"(S (NP.1 Mickey) (VP (V knows) (S (NP.2 Donald) (VP (V owes) (NP.2 himself)))))",\
# NP VP S( PRO VP NP)
"(S (NP.1 Mickey) (VP (V knows) (S (NP.2 he) (VP (V owes) (NP.3 Jack)))))",\
"(S (NP.1 Mickey) (VP (V knows) (S (NP.1 he) (VP (V owes) (NP.3 Jack)))))",\
# NP VP S( PRO VP PRO)
"(S (NP.1 Mickey) (VP (V knows) (S (NP.2 he) (VP (V owes) (NP.3 him)))))",\
"(S (NP.1 Mickey) (VP (V knows) (S (NP.1 he) (VP (V owes) (NP.2 him)))))",\
"(S (NP.1 Mickey) (VP (V knows) (S (NP.2 he) (VP (V owes) (NP.1 him)))))",\
# NP VP S( PRO VP REFL)
"(S (NP.1 Mickey) (VP (V knows) (S (NP.2 he) (VP (V owes) (NP.2 himself)))))",\
"(S (NP.1 Mickey) (VP (V knows) (S (NP.1 he) (VP (V owes) (NP.1 himself)))))",\
# PRO VP S( NP VP NP)
"(S (NP.1 He) (VP (V knows) (S (NP.2 Donald) (VP (V owes) (NP.3 Jack)))))",\
# PRO VP S( NP VP PRO)
"(S (NP.1 He) (VP (V knows) (S (NP.2 Donald) (VP (V owes) (NP.1 him)))))",\
"(S (NP.1 He) (VP (V knows) (S (NP.2 Donald) (VP (V owes) (NP.3 him)))))",\
# PRO VP S( NP VP REFL)
"(S (NP.1 He) (VP (V knows) (S (NP.2 Donald) (VP (V owes) (NP.2 himself)))))",\
# PRO VP S( PRO VP NP)
"(S (NP.1 He) (VP (V knows) (S (NP.2 he) (VP (V owes) (NP.3 Jack)))))",\
"(S (NP.1 He) (VP (V knows) (S (NP.1 he) (VP (V owes) (NP.3 Jack)))))",\
# PRO VP S( PRO VP PRO)
"(S (NP.1 He) (VP (V knows) (S (NP.2 he) (VP (V owes) (NP.3 him)))))",\
"(S (NP.1 He) (VP (V knows) (S (NP.1 he) (VP (V owes) (NP.2 him)))))",\
"(S (NP.1 He) (VP (V knows) (S (NP.2 he) (VP (V owes) (NP.1 him)))))",\
# PRO VP S( PRO VP REFL)
"(S (NP.1 He) (VP (V knows) (S (NP.1 he) (VP (V owes) (NP.1 himself)))))",\
"(S (NP.1 He) (VP (V knows) (S (NP.2 he) (VP (V owes) (NP.2 himself)))))",\

#		Subordinate Clauses
# S( SUB(S(NP VP)) S(NP VP)
#	"(S (SUB While (S (NP.1 Steve) (VP (V slept)))) (S (NP.2 Jose) (VP (V snored))))",\
# S( SUB(S(NP VP)) S(PRO VP)
#	"(S (SUB While (S (NP.1 Steve) (VP (V slept)))) (S (NP.1 he) (VP (V snored))))",\
#	"(S (SUB While (S (NP.1 Steve) (VP (V slept)))) (S (NP.2 he) (VP (V snored))))",\
# S( SUB(S(PRO VP)) S(NP VP)
#	"(S (SUB While (S (NP.1 he) (VP (V slept)))) (S (NP.1 Steve) (VP (V snored))))",\
#	"(S (SUB While (S (NP.1 he) (VP (V slept)))) (S (NP.2 Steve) (VP (V snored))))",\
# S( SUB(S(PRO VP)) S(PRO VP)
#	"(S (SUB While (S (NP.1 he) (VP (V slept)))) (S (NP.1 he) (VP (V snored))))",\
#	"(S (SUB While (S (NP.1 he) (VP (V slept)))) (S (NP.2 he) (VP (V snored))))",\
