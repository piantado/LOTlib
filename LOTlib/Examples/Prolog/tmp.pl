

:- discontiguous(female/1).
:- discontiguous(male/1).
:- discontiguous(parent/2).
:- discontiguous(grandparent/2).
:- style_check(-singleton).

spouse(barak, michelle).
male(barak).
female(michelle).
parent(michelle, sasha).
parent(michelle, malia).
parent(barak, sasha).
parent(barak, malia).
female(sasha).
female(malia).

parent(baraksr, barak).
parent(ann, barak).

parent(hussein, baraksr).
parent(akumu, baraksr).

male(X2) :- male(X2). 

