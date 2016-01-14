

Introduction
============

This contains an example prolog hypothesis for learning the term "grandparent." A hypothesis here consists of a prolog program plus a set of base facts that are always given. 

The project is primarily an example to show how to use a LOTHypothesis to wrap an external interpreter (here: swi-prolog using pyswip).  This interface is still a little buggy -- for reasons I don't understand it is very sensitive to the file name used, and it seems to not want to use a unique temporary file name for each hypothesis. 