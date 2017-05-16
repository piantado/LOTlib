#!/bin/bash

# for running other than bluehive, like compute 

for lang in XXR Fibo XX AmBnCmDn AnBnCn AnCmBn AB ABn An AnB2n AnBn Dyck AnBm Man Reber Saffran BerwickPilato
do
   python plot.py --language=$lang
done
