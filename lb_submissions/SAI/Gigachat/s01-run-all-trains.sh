rm -f benchmarks.log

python rumed_nli.py --answer-mode='v0' --path-in='MedBench/RuMedNLI/dev.jsonl'
python rumed_nli.py --answer-mode='v1' --path-in='MedBench/RuMedNLI/dev.jsonl'
python rumed_nli.py --answer-mode='v2' --path-in='MedBench/RuMedNLI/dev.jsonl'

python rumed_test.py --path-in="RuMedTest--sogma--dev.jsonl" --answer-mode=v0
python rumed_test.py --path-in="RuMedTest--sogma--dev.jsonl" --answer-mode=v1
python rumed_test.py --path-in="RuMedTest--sogma--dev.jsonl" --answer-mode=v2
python rumed_test.py --path-in="RuMedTest--sogma--dev.jsonl" --answer-mode=v3
python rumed_test.py --path-in="RuMedTest--sogma--dev.jsonl" --answer-mode=v4
python rumed_test.py --path-in="RuMedTest--sogma--dev.jsonl" --answer-mode=v5
python rumed_test.py --path-in="RuMedTest--sogma--dev.jsonl" --answer-mode=v6

python rumed_da_net.py --path-in='MedBench/RuMedDaNet/dev.jsonl'
# python rumed_da_net.py --path-in='MedBench/RuMedDaNet/train.jsonl'

echo 'Benchmarks:'
cat benchmarks.log
