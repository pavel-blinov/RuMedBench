#!/bin/bash

python -u double_text_classifier.py --task-name 'RuMedNLI'
python -u double_text_classifier.py --task-name 'RuMedDaNet'
python -u test_solver.py --task-name 'RuMedTest'

zip -m tfidf.zip RuMedDaNet.jsonl RuMedNLI.jsonl RuMedTest.jsonl
