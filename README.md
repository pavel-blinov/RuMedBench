![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# RuMedBench
A **Ru**ssian **Med**ical language understanding **Bench**mark is the set of NLP tasks on medical textual data for the Russian language.

This repository contains code and data to reproduce the results of the paper [*RuMedBench: A Russian Medical Language Understanding Benchmark*](https://arxiv.org/abs/2201.06499).

[Video from the AIME 2022 conference](https://youtu.be/ZO7BoRzJdmE)

## Tasks Descriptions
- **RuMedTop3**\* is the task for diagnosis prediction from a raw medical text, including patient symptoms and complaints.

- **RuMedSymptomRec**\* Given an incomplete medical text, the task is to recommend the best symptom to check or verify.

- **RuMedDaNet** is the yes/no question answering task in the range of medical-related domains (pharmacology, anatomy, therapeutic medicine, etc).

- **RuMedNLI** is the natural language inference task in the clinical domain. The data is the full translated counterpart of [MedNLI](https://jgc128.github.io/mednli/) data.

- **RuMedNER** is the task of named entity recognition in drug-related user reviews. The data is from the [RuDReC](https://github.com/cimm-kzn/RuDReC) repo.

- **ECG2Pathology** is the task of estimation the quality of multilabel-classification on ECG signals from the [PTB-XL dataset](https://physionet.org/content/ptb-xl/).

\*Both tasks are based on the [RuMedPrime](https://zenodo.org/record/5765873#.YbBlXT9Bzmw) dataset.

## Baselines & Results
We have implemented several baseline models; please see details in the paper.

*Accuracy* is the base metric for all tasks evaluation. For some tasks, additional metrics are used:
- **RuMedTop3** and **RuMedSymptomRec** - *Hit@3*
- **RuMedNER** - *F1-score*

Test results for the NLP-tasks:

| Model | RuMedTop3 | RuMedSymptomRec | RuMedDaNet | RuMedNLI | RuMedNER | Overall |
| ------ | :------: | :------: | :------: | :------: | :------: | :------: |
|Naive | 10.58/22.02 | 1.93/5.30 | 50.00 | 33.33 | 93.66/51.96 | 35.21 |
|Feature-based | **49.76/72.75**  |  32.05/49.40  |   51.95 |  59.70  | 94.40/62.89 |  58.46  |
|BiLSTM | 40.88/63.50  |  20.24/31.33  |   52.34 |  60.06  | 94.74/63.26 |  53.87  |
|RuBERT | 39.54/62.29 | 18.55/34.22 | 67.19 | 77.64 | 96.63/73.53 | 61.44 |
|RuPoolBERT | 47.45/70.44 | 34.94/52.05 | 71.48 | 77.29 | 96.47/73.15 | 67.20 |
|RuBioBERT* | 43.55/68.86 | 28.94/44.55 | 53.91 | 80.31 | 96.63/75.97 | 62.69 |
|RuBioRoBERTa* | 46.72/72.87 | **44.01/58.95** | 76.17 | 82.77 | **97.19/77.81** | **71.54** |
|Human | 25.06/48.54 | 7.23/12.53 | **93.36** | **83.26** | 96.09/76.18 | 61.89 |

Test results for the ECG2Pathology task:

| Model | Macro-F1 |
| ------ | :------: |
|Human | 39.34 |
|Naive | 1.15 |

We define the overall model score as mean over all metric values (with prior averaging in the case of two metrics).

\* this is implementation from the paper [RuBioRoBERTa: a pre-trained biomedical language model for Russian language biomedical text mining](https://arxiv.org/abs/2204.03951) ([repository](https://github.com/alexyalunin/RuBioRoBERTa)).

You can find the extension of this benchmark (with closed test sets) on the [MedBench platform](https://medbench.ru/).

## How to Run
Please refer to the [`code/`](code/) directory.

## Contact
If you have any questions, please post a Github issue or email the authors.

## Citation
```bibtex
@misc{blinov2022rumedbench,
    title={RuMedBench: A Russian Medical Language Understanding Benchmark},
    author={Pavel Blinov and Arina Reshetnikova and Aleksandr Nesterov and Galina Zubkova and Vladimir Kokh},
    year={2022},
    eprint={2201.06499},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
