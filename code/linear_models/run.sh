#!/bin/bash

out=$(pwd)'/out'
mkdir -p $out

python -u single_text_classifier.py --task-name 'RuMedTop3'
python -u single_text_classifier.py --task-name 'RuMedSymptomRec'
python -u double_text_classifier.py --task-name 'RuMedNLI'
python -u double_text_classifier.py --task-name 'RuMedDaNet'
python -u token_classifier.py --task-name 'RuMedNER'
