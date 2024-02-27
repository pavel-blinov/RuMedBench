## Dependencies and Library Versions
For specific lib versions, see `requirements.txt` and install them with 
```bash
pip install -r requirements.txt
```

## Hardware Requirements
The code runs on:
```
CPU Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
GPU NVIDIA Tesla V100-PCIE-16GB
RAM 16GB
HDD 16GB
```

## General Description
Each directory `bert`, `bilstm` and `linear_models` contains a baseline model.

Generally, a model should produce output directory (e.g. `bert/out`) with result `jsonl` files named after the task name (e.g. `RuMedTop3.jsonl`).

Each file contains same samples as in test parts enhanced with the `prediction` field.<br/>
Examples,<br/>
for `RuMedTop3.jsonl`
```
{
    "idx": "qaf1454f",
    "code": "I11",
    "prediction": ["I11", "I20", "I10"]
}
```

or `RuMedSymptomRec.jsonl`
```
{
    "idx": "q45f6321",
    "code": "боль в шее",
    "prediction": ["тошнота", "боль в шее", "частые головные боли"]
}
```

or `RuMedDaNet.jsonl`
```
{
    "pairID": "f5309eadb4eacf0f144b24e260643ea2",
    "answer": "да",
    "prediction": "нет"
}
```

or `RuMedNLI.jsonl`
```
{
    "pairID": "1f2a8146-66c7-11e7-b4f2-f45c89b91419",
    "gold_label": "entailment",
    "prediction": "neutral"
}
```

or `RuMedNER.jsonl`
```
{
    "idx": "769708.tsv_5",
    "ner_tags": ["B-Drugname", "O", "B-Drugclass", "O", "O"],
    "prediction": ["B-Drugclass", "O", "O", "O", "O"]
}
```

### tasks_builder.py

It is the script used to prepare data for the benchmark tasks from raw data files.

```bash
python tasks_builder.py
```

### eval.py

It is the script to evaluate the test results.

Run it like 
```bash
python eval.py --out_dir bert/out
```
or
```bash
python eval.py --out_dir human
```
