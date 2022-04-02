To run the BERT models:
1) Download the [RuBERT model](http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_pt.tar.gz) & extract it to `models/rubert_cased_L-12_H-768_A-12_pt`.
```bash
mkdir -p models/; cd models/
wget "http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_pt.tar.gz"
tar -xvzf rubert_cased_L-12_H-768_A-12_pt.tar.gz
```
2) Run<br/>
`./run.sh bert` for *RuBERT* model<br/>
or<br/>
`./run.sh pool` for *RuPoolBERT* model.