#!/bin/bash

python -u double_text_classifier.py --task-name 'RuMedDaNet' --device 2
python -u double_text_classifier.py --task-name 'RuMedNLI' --device 2
python -u test_solver.py --task-name 'RuMedTest'

zip -m rnn.zip RuMedDaNet.jsonl RuMedNLI.jsonl RuMedTest.jsonl
