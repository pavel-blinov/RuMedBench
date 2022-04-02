This directory contains feature-based models (logistic regression model with tf-idf vectorizer and CRF).

### How to run

`./run.sh` or you can run the model for different tasks separately, e.g.

```bash
python single_text_classifier.py --task-name='RuMedSymptomRec'
```

The models produce results in `.jsonl` format to output directory `out`.
