This directory contains BiLSTM model with randomly initialized embeddings.

### How to run

`./run.sh` or you can run the model for different tasks separately, e.g.

```bash
python single_text_classifier.py --task-name='RuMedSymptomRec' --device=0
```

The models produce results in `.jsonl` format to output directory `out`.
