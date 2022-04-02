#!/bin/bash

type=$@

out=$(pwd)'/out'
mkdir -p $out

# run the tasks sequentially
python -u single_text_classifier.py --gpu 0 --task_name 'RuMedTop3' --bert_type $type
python -u single_text_classifier.py --gpu 0 --task_name 'RuMedSymptomRec' --bert_type $type
python -u double_text_classifier.py --gpu 0 --task_name 'RuMedDaNet' --bert_type $type
python -u double_text_classifier.py --gpu 0 --task_name 'RuMedNLI' --bert_type $type
python -u token_classifier.py --gpu 0 --task_name 'RuMedNER' --bert_type $type

# # or run in parallel on multiple gpus
# python -u single_text_classifier.py --gpu 0 --task_name 'RuMedTop3' --bert_type $type &
# python -u single_text_classifier.py --gpu 1 --task_name 'RuMedSymptomRec' --bert_type $type &
# python -u double_text_classifier.py --gpu 2 --task_name 'RuMedDaNet' --bert_type $type &
# python -u double_text_classifier.py --gpu 3 --task_name 'RuMedNLI' --bert_type $type &
# wait
# python -u token_classifier.py --gpu 0 --task_name 'RuMedNER' --bert_type $type
