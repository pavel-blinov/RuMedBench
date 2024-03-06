mkdir -p 'out'

# python rumed_nli.py --path-in='MedBench/RuMedNLI/test.jsonl' --path-out='out/medbench--rumednli--v0.jsonl' --answer-mode='v0' 
# python rumed_nli.py --path-in='MedBench/RuMedNLI/test.jsonl' --path-out='out/medbench--rumednli--v1.jsonl' --answer-mode='v1' 
python rumed_nli.py --path-in='../../../data/RuMedNLI/test.jsonl' --path-out='out/RuMedNLI.jsonl' --answer-mode='v2'

# python rumed_test.py --path-in='MedBench/RuMedTest/test.jsonl' --path-out='out/medbench--rumedtest--v2.jsonl' --answer-mode=v2
# python rumed_test.py --path-in='MedBench/RuMedTest/test.jsonl' --path-out='out/medbench--rumedtest--v3.jsonl' --answer-mode=v3
# python rumed_test.py --path-in='MedBench/RuMedTest/test.jsonl' --path-out='out/medbench--rumedtest--v4.jsonl' --answer-mode=v4
python rumed_test.py --path-in='../../../data/RuMedTest/test.jsonl' --path-out='out/RuMedTest.jsonl' --answer-mode=v6

python rumed_da_net.py --path-in='../../../data/RuMedDaNet/test.jsonl' --path-out='out/RuMedDaNet.jsonl'
