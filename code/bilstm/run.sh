#!/bin/bash

out=$(pwd)'/out'
mkdir -p $out

python -u single_text_classifier.py --task-name 'RuMedTop3' --device 0
python -u single_text_classifier.py --task-name 'RuMedSymptomRec' --device 0
python -u double_text_classifier.py --task-name 'RuMedDaNet' --device 0
python -u double_text_classifier.py --task-name 'RuMedNLI' --device 0
python -u token_classifier.py --task-name='RuMedNER' --device 0
