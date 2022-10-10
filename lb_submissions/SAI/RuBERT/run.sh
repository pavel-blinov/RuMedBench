#!/bin/bash

type=$@

models=$(pwd)'/models'
mkdir -p $models;

if [ ! -f $models'/rubert_cased_L-12_H-768_A-12_pt/pytorch_model.bin' ]; then
    echo $models'/rubert_cased_L-12_H-768_A-12_pt/pytorch_model.bin'
    cd $models;
    wget "http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_pt.tar.gz"
    tar -xvzf rubert_cased_L-12_H-768_A-12_pt.tar.gz
    cd ../;
fi

python -u double_text_classifier.py --task-name 'RuMedDaNet' --device 0 --bert-type $type
python -u double_text_classifier.py --task-name 'RuMedNLI' --device 0 --bert-type $type
python -u test_solver.py --task-name 'RuMedTest' --device 0 --bert-type $type

zip -m $type.zip RuMedDaNet.jsonl RuMedNLI.jsonl RuMedTest.jsonl
