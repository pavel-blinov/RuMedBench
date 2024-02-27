"""Checking RumedDaNet."""

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from rumed_utils import log_answer, parse_element, run_main, wrapped_fire

PROMPT = """Ты доктор и должен пройти экзамен. Тебе даны два утверждения.
Первое -- абсолютно верное и должно быть базой для твоего ответа: "{ru_sentence1}".

Второе утверждение таково: "{ru_sentence2}".

Ты должен ответить, чем является второе утверждение в контексте первого:
1. Следствие
2. Противоречие
3. Нейтральность.

Ответь одним словом:""".strip()

PROMPT_2_COT = """Ты доктор и должен пройти экзамен. Тебе даны два утверждения.
Первое -- абсолютно верное и должно быть базой для твоего ответа: "{ru_sentence1}".

Второе утверждение таково: "{ru_sentence2}".

Вопрос: чем является второе утверждение в контексте первого?
1. Следствие
2. Противоречие
3. Нейтральность.

Рассуждай шаг за шагом и выбери правильный вариант.
"""

LABELS_MAP = {
    'следствие': 'entailment',
    'противоречие': 'contradiction',
    'нейтральность': 'neutral',
}

POSSIBLE_ANSWERS = {'следствие', 'противоречие', 'нейтральность'}

def get_answer_v0(llm, q_dict):
    input_message = PROMPT.format(**q_dict)
    llm_response = llm.invoke(input_message).content
    answer = parse_element(llm_response, POSSIBLE_ANSWERS)
    if answer == '2. Противоречие':
        answer = 'противоречие'
    elif answer == '3. Нейтральность.':
        answer = 'нейтральность'
    elif answer == '2.':
        answer = 'противоречие'
    answer = LABELS_MAP.get(answer)
    true_answer = q_dict.get('gold_label')
    possible_answers = LABELS_MAP.values()
    log_answer(q_dict, possible_answers, true_answer, input_message, llm_response, answer)
    return answer

def get_answer_v1(llm, q_dict):
    input_message = PROMPT_2_COT.format(**q_dict)
    doctor_answer = llm.invoke(input_message).content
    possible_answers_pretty = '{' + ', '.join(POSSIBLE_ANSWERS) + '}'
    ru_sentence1 = q_dict['ru_sentence1']
    ru_sentence2 = q_dict['ru_sentence2']
    moderator_msg = f'''Ниже представлены вопрос из теста, а также ответ на этот вопрос со стороны врача. Врач отвечает развёрнуто. Твоя задача -- понять, какой же именно вариант ответа из {possible_answers_pretty} выбрал врач.
=====
Задача:
Даны два утверждения.
Первое -- абсолютно верное и должно быть базой для твоего ответа: "{ru_sentence1}".

Второе утверждение таково -- "{ru_sentence2}".

Ты должен ответить, чем является второе утверждение в контексте первого:
1. Следствие
2. Противоречие
3. Нейтральность.
=====
Ответ врача:
{doctor_answer}
=====
Ответь одним словом из {possible_answers_pretty}:
'''
    moderator_answer = llm.invoke(moderator_msg).content
    answer = parse_element(moderator_answer, POSSIBLE_ANSWERS)
    answer = LABELS_MAP.get(answer)
    true_answer = q_dict.get('gold_label')
    possible_answers = LABELS_MAP.values()
    log_answer(q_dict, possible_answers, true_answer, moderator_msg, moderator_answer, answer)
    return answer

ASK_ATTEMPTS = 3
def get_answer_v2(llm, q_dict):
    input_message = PROMPT_2_COT.format(**q_dict)
    possible_answers_pretty = '{' + ', '.join(POSSIBLE_ANSWERS) + '}'

    system_msg = SystemMessage(content=input_message)
    memory = [system_msg]
    for at in range(ASK_ATTEMPTS):
        ai_msg = llm.invoke(memory)
        text = ai_msg.content
        answer = parse_element(text, POSSIBLE_ANSWERS)
        answer = LABELS_MAP.get(answer)
        if answer:
            break
        memory.append(ai_msg)
        memory.append(HumanMessage(content=f'Ответь одним словом из {possible_answers_pretty}.'))
    true_answer = q_dict.get('gold_label')
    possible_answers = LABELS_MAP.values()
    moderator_msg = input_message
    moderator_answer = '{\n' + '\n###\n'.join(msg.content for msg in memory[1:]) + '\n}'
    log_answer(q_dict, possible_answers, true_answer, moderator_msg, moderator_answer, answer)
    return answer

get_answer_map = {
    'v0': get_answer_v0,
    'v1': get_answer_v1,
    'v2': get_answer_v2,
}

def answer_to_test_output(q_dict, answer) -> dict:
    return {'pairID': q_dict['pairID'], 'gold_label': answer}

def main(path_in, config_path='config.ini', answer_mode='v0', path_out=None):
    run_main(
        path_in=path_in,
        config_path=config_path,
        path_out=path_out,
        get_answer_map=get_answer_map,
        answer_mode=answer_mode,
        answer_field='gold_label',
        answer_to_test_output=answer_to_test_output,
    )

if __name__ == '__main__':
    wrapped_fire(main)
