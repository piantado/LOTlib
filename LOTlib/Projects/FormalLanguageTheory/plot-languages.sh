#!/bin/bash

# for running other than bluehive, like compute 

for lang in XXR Fibo XX AmBnCmDn AnBnCn AnCmBn AB ABn An AnB2n AnBn Dyck AnBm Man Reber Saffran BerwickPilato XY  Gomez2 Gomez6 Gomez12 NewportAslin MorganNewport
do
   echo "# Plotting language " $lang
   python plot.py --language=$lang
done
