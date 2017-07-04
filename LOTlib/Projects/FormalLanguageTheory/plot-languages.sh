#!/bin/bash

# for running other than bluehive, like compute 

for lang in  XXR Fibo XX XY AmBnCmDn AnBnCn AnCmBn AB ABn An AnB2n AnBn Dyck AnBm Man Reber Saffran BerwickPilato Gomez2 Gomez6 Gomez12 NewportAslin MorganNewport MorganMeierNewport HudsonKamNewport100 HudsonKamNewport75 HudsonKamNewport60 HudsonKamNewport45 ReederNewportAslin SimpleEnglish  AAAA AnBmCmAn ABA ABB
do
   echo "# Plotting language " $lang
   python plot.py --language=$lang &
done
