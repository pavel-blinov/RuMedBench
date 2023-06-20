Each task directory (starting with *RuMed*\*) contains `train/dev/test` data files in `jsonl`-format.

### RuMedTop3
```
{
    "idx": "qd4405c5",
    "symptoms": "Сердцебиение, нарушение сна, ощущение нехватки воздуха.
    Боль и хруст в шеи, головные боли по 3 суток подряд.",
    "code": "M54"
}
```

### RuMedSymptomRec
```
{
    "idx": "qbaecae4",
    "symptoms": "пациентка на приеме с родственниками. Со слов родственников - жалобы на плохой сон,
    чувство страха, на навязчивые мысли,что 'ее кто-то бьет'",
    "code": "колебания артериального давления"
}
```

### RuMedDaNet
```
{
    "pairID": "b2d69800b0a141aa63bd1104c6d53488",
    "context": "Эпилепсия — хроническое полиэтиологическое заболевание головного мозга, доминирующим
     проявлением которого являются повторяющиеся эпилептические припадки, возникающие вследствие
     усиленного гиперсинхронного разряда нейронов головного мозга.",
    "question": "Эпилепсию относят к заболеваниям головного мозга человека?",
    "answer": "да",
}
```

### RuMedNLI
```
{
    "pairID": "1892e470-66c7-11e7-9a53-f45c89b91419",
    "ru_sentence1": "Во время госпитализации у пациента постепенно усиливалась одышка, что потребовало
    выполнения процедуры неинвазивной вентиляции лёгких с положительным давлением, а затем маска без ребризера.",
    "ru_sentence2": "Пациент находится при комнатном воздухе.",
    "gold_label": "contradiction",
}
```

### RuMedNER
```
{
    "idx": "769708.tsv_5",
    "tokens": ["Виферон", "обладает", "противовирусным", "действием", "."],
    "ner_tags": ["B-Drugname", "O", "B-Drugclass", "O", "O"]
}
```

### ECG2Pathology
```
{
    "record_name": "00009_hr",
    "age": 55.0,
    "sex": 0,
    ...,
    "targets": [37,54]
}
```

<details>
  <summary>raw</summary>

The directory contains raw data files.

The tasks `RuMedTop3` and `RuMedSymptomRec` are based on the [`RuMedPrime`](https://zenodo.org/record/5765873#.YbBlXT9Bzmw) dataset.
The file `RuMedPrimeData.tsv` contains:
```
symptoms    anamnesis   icd10   new_patient_id  new_event_id    new_event_time
Сухость кожи... Месяц назад...  E01.8   qf156c36    q5fc2cb1    2027-05-19
Жалобы ГБ...    Начало острое...    J06.9   q9321cf8    qe173f20    2023-03-24
```
- `symptoms` is the text field with patient symptoms and complaints;
- `icd10` - ICD-10 disease code;
- `new_event_id` is the sample id.

The file `rec_markup.csv` contains markup for the recommendation task:
```
new_event_id,code,keep_spans
q5fc2cb1,"кожа, сухая","[(0, 0), (7, 12), (13, 108)]"
qe173f20,боль в мышцах,"[(0, 138), (151, 279)]"
q653efaa,боль в мышцах,"[(0, 57), (70, 129)]"
qe48681b,боль жгучая,"[(0, 45), (56, 181)]"
```
- `new_event_id` is the sample id;
- `code` is a symptom to predict;
- `keep_spans` is the list of `(start, end)` tuples, as we neet to transform the original text to exclude target symptom-code.

The data `RuMedNLI` is based on translated [MedNLI](https://jgc128.github.io/mednli/) data.

> Important! This repository do not contain RuMedNLI files, please download them (`ru_mli_train_v1.jsonl`, `ru_mli_dev_v1.jsonl` and `ru_mli_test_v1.jsonl`) from [RuMedNLI: A Russian Natural Language Inference Dataset For The Clinical Domain](https://doi.org/10.13026/gxzd-cf80) to the `raw` directory. Then run `python tasks_builder.py` from the `code/` directory.

The task `RuMedNER` is based on RuDReC data - https://github.com/cimm-kzn/RuDReC.
`RuDReC.csv` is the dataframe file with named entities in [IOB format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)).
```
Sentence#,Word,Tag
172744.tsv_0,нам,O
172744.tsv_0,прописали,O
172744.tsv_0,",",O
172744.tsv_0,так,O
172744.tsv_0,мой,O
172744.tsv_0,ребенок,O
172744.tsv_0,сыпью,B-ADR
172744.tsv_0,покрылся,I-ADR
172744.tsv_0,",",O
```
</details>
