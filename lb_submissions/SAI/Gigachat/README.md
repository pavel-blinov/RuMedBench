# Testing MedBench with Gigachat

## Summary

We tested the performance of the Gigachat (`GigachatPro, uncensored, 2024-03-04`) model on RuMedBench tasks with the following results:

| Task       | Result |
|------------|--------|
| RuMedNLI   |`65.17%`|
| RuMedDaNet |`92.58%`|
| RuMedTest  |`72.04%`|

## Experiments description
### RuMedDaNet ( `rumed_da_net.py` )

Only one simple prompt was used -- just `{context}`, `{question}` and request to answer "Yes" or "No".

**Accuracy (dev)**: `95.70 %`.
### RuMedNLI ( `rumed_nli.py` )
3 approaches was used:
0. **Simple doctor prompt**: one prompt (with doctor role description, instruction and request to __respond in one word__) sent to LLM.
1. **Complex doctor prompt with moderator**: one prompt (with doctor role description, instruction and request to __respond in details__) sent to LLM. Then another one prompt (with moderator role description, request to choose the right answer and to respond in one word) sent to LLM.
2. **Complex doctor prompt with chat**: one prompt (with doctor role description, instruction and request to __respond in details__) sent to LLM. Then, if response isn't specific, new request with chat history sent to LLM. (only 3 times maximum)

**Accuracy (dev)**:

| Approach            | Accuracy  |
|---------------------|-----------|
| v0: simple          | `60.55 %` |
| v1: doctor + prompt | `67.51 %` |
| v2: doctor + chat   | `67.93 %` |

Approach `v2` was used for test evaluation.

### RuMedTest ( `rumed_test.py` )
For prompt evaluation sogma dataset was used [link](https://geetest.ru/tests/terapiya_(dlya_internov)_sogma_).

7 experiments were checked:
0. **Simple prompt**: one prompt with question instruction sent to LLM. Invalid answers ignored.
1. **Simple doctor prompt**: one prompt with doctor role, question instruction and request to respond in one number was sent to LLM. Invalid answers ignored.
2. **Complex doctor prompt with moderator**: like approach [RuMedNLI:1]
3. **Complex doctor prompt with moderator (2)**: like previous approach, [RuMedTest:2].
4. **Complex doctor prompt with moderator (3)**: like [RuMedTest:2].
5. **Complex doctor prompt with chat**: like [RuMedNLI:2]
6. **Simple alphabetic doctor prompt with chat**: like previous approach, [RuMedTest:5], but numbers for variants replaced with letters (`1 -> a`, `2 -> b`, etc.)
7. **Complex alphabetic doctor prompt with chat**: like previous approach, [RuMedTest:6], but with instruction to __respond in details__.

**Accuracy (sogma)**:
| Approach | Accuracy  |
|----------|-----------|
| v0       | `49.15 %` |
| v1       | `55.15 %` |
| v2       | `53.46 %` |
| v3       | `53.46 %` |
| v4       | `49.41 %` |
| v5       | `52.93 %` |
| v6       | `57.24 %` |
| v7       | `~26.1 %` |

Approach `v6` was used for test evaluation.
## Usage
0. Put `config.ini` with Gigachat credentials to this directory.

Example of content (without <> brackets):
```ini
[credentials]
user = <your-user>
credentials = <your-credentials>
scope = <your-scope>
[base_url]
base_url = https://developers.sber.ru/...
```

1. Run `s00-prepare.sh` to download tests.
2. Run `s01-run-all-trains.sh` to evaluate train/dev.
3. Run `s02-run-all-tests.sh` to generate test samples.
### Notes
- you can reuse this framework to check other LLMS: replace `rumed_utils#create_llm_gigachat` with something else
- to speed up test rerunning caching used (see `set_llm_cache` in `rumed_utils`)
