# Testing MedBench tasks with ChatGPT

## Summary
We tested the performance of the ChatGPT (proxied by `text-davinci-003`) model on RuMedBench tasks with the following results:
- RuMedTest: 35.0%  (above other models to date, yet way below human level)
- RuMedDaNet: 89.3%  (way above other models to date, closer to human level)
- RuMedNLI: 61.3%  (lagging behind other models to date, way below human level)
While some preliminary tests were done to select better prompting techniques, these results may potentially be improved with more prompt testing and further model fine-tuning.

## Premises
We intended to evaluate the performance of the ChatGPT large language model with RuMedBench questions. The evaluation was done using [ChatGPT web interface](https://chat.openai.com/chat), and then main runs were done with open.ai API with model `text-davinci-003`. The latter is not precisely ChatGPT, but supposedly closely related, and also shown similar answer quality (measured on a sample of 20 questions from RuMedTest). An additional benefit of API access was that `text-davinci-003` was closer to observing answer formats without resorting to longer explanations like ChatGPT sometimes did.

To gain better quality, we compared the performance of smaller samples with similar tasks (dev sample in the case of RuMedDaNet and RuMedNLI, similar medical questions from medical school exams for RuMedTest). The tested approaches included Zero- and One-shot prompts and translation into English. See details of such prompt tests in the individual tests' sections.

All tests were run between 10 and 15 Feb 2023.

## RuMedTest
For prompt evaluation, we used questions from a similar test [found here](https://geetest.ru/tests/terapiya_(dlya_internov)_sogma_). It has 775 questions; among them, only 355 have the 1-line format, similar to RuMedTest. Some minor (~5%) overlap in questions with RuMedTest was noticed. Those questions were excluded from prompt evaluation.

The prompt tests included Zero- and One-shot prompts and translation into English. There was no significant difference observed (perhaps the prompt test sample was too small), so the main benchmark tests were performed using straight prompt in Russian:
```
You are a medical doctor and need to answer which one of the four statements is correct:
1. "Лёгочное сердце" может возникнуть при ишемической болезни сердца".
2. "Лёгочное сердце" может возникнуть при гипертонической болезни".
3. "Лёгочное сердце" может возникнуть при хронической обструктивной болезни лёгких".
4. "Лёгочное сердце" может возникнуть при гипертиреозе".
The number of the correct statements is:
```
The resulting accuracy was 35% which beats simpler models but leaves a large gap with human performance. Possible ways to improve this result with the same model would be more extensive prompt testing (greater variety and larger testing sample), fine-tuning (if a large quantity of similar test data and budget are available), cleaning the test data from typos, rare acronyms and abbreviations.

## RuMedDaNet
Prompt testing was done using both English and Russian, with Zero and one shot, with context and without it. The case with Zero-shot in Russian with included context was performing reasonably well (85% on the pre-test sample) and was chosen for the whole benchmark run.

Note that the model could answer prompts without context significantly better than randomly, at 67%, which indicates some domain knowledge in the model. It may be worth exploring unprompted Yes/No tests more, possibly adding them as a benchmark component.

Example of prompt + question used:
```
Imagine that you are a medical doctor and know everything about medicine and need to pass a degree exam. 
The context is: Природа полос поглощения в ик-области связана с колебательными переходами и изменением колебательных состояний ядер, входящих в молекулу поглощающего вещества. Поэтому поглощением в ИК-области обладают молекулы, дипольные моменты которых изменяются при возбуждении колебательных движений ядер. Область применения ИК-спектроскопии аналогична, но более широка, чем УФ-метода. ИК-спектр однозначно характеризует всю структуру молекулы, включая незначительные ее изменения. Важные преимущества данного метода — высокая специфичность, объективность полученных результатов, возможность анализа веществ в кристаллическом состоянии.
The question is: Возможности ИК-спектроскопии позволяют анализировать вещества в кристаллическом состоянии?
You should answer only yes or no.
The answer is 
```
The resulting accuracy was 89.3% which beats simpler models (best-registered result to date at 68%) and gets close to human performance (93%).

## RuMedNLI
Prompt testing was done using Russian only, with Zero and a few shots. There was no significant difference in performance (0.75 acc in prompt tests); a case with a few-shot in Russian was chosen for the whole benchmark run.

Example of the prompt:
```
You are a medical doctor and need to pass a degree exam. You are given two statements and need to answer how the second statement relates to the first statement. Possible answers are 'entailment', 'contradiction', or 'neutral'
Statement 1: "В анамнезе нет тромбозов или ТГВ, никогда не было болей в груди до случаев недельной давности."
Statement 2: "Пациент страдает стенокардией"
Answer: "entailment"
Statement 1: "В течение последней недели стал более сонливым и трудно возбудимым."
Statement 2: "В последнюю неделю он был менее внимателен"
Answer: "entailment"
Statement 1: "КТ головы показала небольшое правое височное внутрипаренхиматозное кровоизлияние 2х2 см, на повторной КТ головы осталось без изменений."
Statement 2: "у пациента было гипертоническое кровотечение"
Answer: "neutral"
Statement 1: "Рентгенограмма чистая, не похоже наличие инфекции."
Statement 2: "Рентген грудной клетки выявил инфильтраты"
Answer: "contradiction"
Statement 1: "КТ головы показала небольшое правое височное внутрипаренхиматозное кровоизлияние 2х2 см, на повторной КТ головы осталось без изменений."
Statement 2: "у пациента была нормальная КТ головы"
Answer: "contradiction"
Statement 1: "Сахарный диабет II типа."
Statement 2: "Инсулин не требуется."
Answer:
```
The resulting accuracy was 61.3%, which does not beat simpler models (67% best to date), not to speak about the human level (85.7%). This could be caused by the fact that NLI tasks are more sensitive to fine-tuning. Also, different prompt techniques may be required.

### An interesting case of disagreement:
When testing NLI and other questions with ChatGPT, we observed cases where the model gave incorrect (vs. benchmark ground truth) answers supported by a detailed explanation. Here is an example:
```
pairID                       18935380-66c7-11e7-9dbb-f45c89b91419
Gold label = CONTRADICTION

PROMPT:
You are a medical doctor and need to pass exam. You are given two statements:
The first statement is absolutely correct and should be the basis for your answer: "На [**2810-12-14**] также был удален желудочковый дренаж."
The second statement is "Отсутствие в анамнезе неврологических симптомов". 
You should answer if the second statement is 'entailment', 'contradiction', or 'neutral'.
The answer is 

The relationship between the two statements is neutral.

CHAT_GPT:
The first statement describes a surgical procedure that was performed on a specific date to remove a cardiac drain. The second statement mentions the absence of neurological symptoms in the patient's medical history. While the two statements may be related in some way, they do not necessarily entail or contradict each other.

The absence of neurological symptoms in a patient's medical history does not necessarily have a direct relationship with the removal of a cardiac drain. The relationship between the two statements may be influenced by other factors such as the patient's overall health, the reason for the drain placement, and the specifics of the surgical procedure. Without further information, it is not possible to determine a definite relationship between the two statements.
```
It may be worth exploring such cases with domain experts to ensure that test questions reflect modern medical knowledge.

### Files descriptions
`chat-rmb.ipynb` - the main codebase for interaction with the model
`*.jsonl` - answer files
`rm*` - files with intermediate data and logs
