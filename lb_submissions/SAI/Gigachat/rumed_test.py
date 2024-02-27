"""Checking RuMedTest."""

import re

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from rumed_utils import log_answer, parse_element, run_main, wrapped_fire

def extract_answer_keys(q_dict):
    snums = map(str, range(1, 10))
    return [si for si in snums if si in q_dict]

def make_input_message_v0(q_dict):
    answers = extract_answer_keys(q_dict)
    prompt = '\n'.join([
        'Выбери номер наиболее корректного утверждения:',
        *('%s. {0} {%s}.' % (si, si) for si in answers),
        '\nНомер наиболее корректного утверждения:',
    ])
    input_message = prompt.format(q_dict['question'], *(q_dict[ii] for ii in answers))
    return input_message

def make_input_message_v1(q_dict):
    answers = extract_answer_keys(q_dict)
    prompt = '\n'.join([
        'Ты врач, сдаёшь медицинский экзамен. Тебе нужно дать правильный ответ на вопрос:\n{0}\n\nВарианты ответа:',
        *('%s. {%s}.' % (si, si) for si in answers),
        '\nКакой из ответов {0} наиболее корректен? Обязательно ответь одним числом.'.format(', '.join(answers)),
    ])
    input_message = prompt.format(q_dict['question'], *(q_dict[ii] for ii in answers))
    return input_message

def make_input_message_v2(q_dict):
    answers = extract_answer_keys(q_dict)
    prompt = '\n'.join([
        'Ты врач, сдаёшь медицинский экзамен. Тебе нужно дать правильный ответ на вопрос:\n{0}\n\nВарианты ответа:',
        *('%s. {%s}.' % (si, si) for si in answers),
        '\nКакой из ответов {0} наиболее корректен? Ответь и объясни, почему'.format(', '.join(answers)),
    ])
    input_message = prompt.format(q_dict['question'], *(q_dict[ii] for ii in answers))
    return input_message

def make_input_message_v3(q_dict):
    answers = extract_answer_keys(q_dict)
    prompt = '\n'.join([
        'Ты врач, сдаёшь медицинский экзамен. Тебе нужно дать правильный ответ на вопрос:\n{0}\n\nВарианты ответа:',
        *('%s. {%s}.' % (si, si) for si in answers),
        '\nРассуждай шаг за шагом и скажи, какой из ответов {0} наиболее корректен? Помни, что правильный ответ только один!'.format(', '.join(answers)),
    ])
    input_message = prompt.format(q_dict['question'], *(q_dict[ii] for ii in answers))
    return input_message

def make_input_message_v4(q_dict):
    answers = extract_answer_keys(q_dict)
    prompt = '\n'.join([
        'Ты сдаёшь тест с одним правильным ответом. Вопрос:\n{0}',
        *('%s. {%s}.' % (si, si) for si in answers),
        '\nРассуждай шаг за шагом и скажи, какой из ответов {0} правильный? Правильный ответ только один!'.format(', '.join(answers)),
    ])
    input_message = prompt.format(q_dict['question'], *(q_dict[ii] for ii in answers))
    return input_message

def get_answer_basic(llm, q_dict, message_maker):
    possible_answers = extract_answer_keys(q_dict)
    input_message = message_maker(q_dict)
    llm_response = llm.invoke(input_message).content
    answer = parse_element(llm_response, possible_answers)
    true_answer = q_dict.get('answer')
    log_answer(q_dict, possible_answers, true_answer, input_message, llm_response, answer)
    return answer

def get_answer_via_roles(llm, q_dict, message_maker):
    input_message = message_maker(q_dict)
    possible_answers = extract_answer_keys(q_dict)
    possible_answers_pretty = '{' + ', '.join(possible_answers) + '}'
    question = q_dict['question']
    variants_fmt = '\n'.join('%s. {%s}' % (si, si) for si in possible_answers)
    variants = '\n'.join('{}. {}'.format(ii, q_dict[ii]) for ii in possible_answers)
    full_answer = llm.invoke(input_message).content
    moderator_msg = f'''Ниже представлены тест в виде вопроса и вариантов ответа, а также ответ на этот вопрос со стороны врача. Врач отвечает развёрнуто. Твоя задача -- понять, какой же именно вариант ответа из {possible_answers_pretty} выбрал врач.
=====
Вопрос:
{question}

Варианты ответа:
{variants}
=====
Ответ врача:
{full_answer}
=====
Ответь одним числом из {possible_answers_pretty}:
'''
    llm_response = llm.invoke(moderator_msg).content
    answer = parse_element(llm_response, possible_answers)
    true_answer = q_dict.get('answer')
    log_answer(q_dict, possible_answers, true_answer, moderator_msg, llm_response, answer)
    return answer

ASK_ATTEMPTS = 5
def get_answer_v5(llm, q_dict):
    possible_answers = extract_answer_keys(q_dict)
    possible_answers_pretty = '[' + ', '.join(possible_answers) + ']'
    input_prompt = '\n'.join([
        'Ты врач, сдаёшь тест. Вопрос:\n{0}',
        *('%s. {%s}.' % (si, si) for si in possible_answers),
        f'\nРассуждай шаг за шагом и скажи, какой из ответов {possible_answers_pretty} правильный? Если правильных ответов несколько выбери один, самый правдоподобный.',
    ])
    input_message = input_prompt.format(q_dict['question'], *(q_dict[ii] for ii in possible_answers))
    memory = [SystemMessage(content=input_message)]
    for at in range(ASK_ATTEMPTS):
        ai_msg = llm.invoke(memory)
        text = ai_msg.content
        answer = parse_element(text, possible_answers)
        if answer:
            break
        memory.append(ai_msg)
        if re.findall('^\d+$', ai_msg.content):
            parts = ', '.join([f'либо {si}' for si in ai_msg.content])
            hc = f'Правильный ответ только один. Остальные неверные. Выбери тот, который тебе кажется наиболее похожим на правду: {parts}'
        else:
            hc = f'Ответь одним числом из {possible_answers_pretty}. Если ты думаешь, что правильных ответов несколько, выбери один, самый правдоподобный'
        memory.append(HumanMessage(content=hc))
    true_answer = q_dict.get('answer')
    moderator_msg = input_message
    moderator_answer = '{\n' + '\n###\n'.join(msg.content for msg in memory[1:]) + '\n}'
    log_answer(q_dict, possible_answers, true_answer, moderator_msg, moderator_answer, answer)
    return answer


VS = 'abcdef'
# VS = 'alpha beta gamma delta epsilon zeta'.split() # works, but slightly worse
ALPHA_MAPPER = dict(zip(map(str, range(1, 7)), VS))
ALPHA_INV_MAPPER = {val:key for key, val in ALPHA_MAPPER.items()}

def get_answer_v6(llm, q_dict):
    possible_answers = extract_answer_keys(q_dict)
    pretty_answers = [ALPHA_MAPPER[an] for an in possible_answers]
    prompt = '\n'.join([
        'Ты врач, сдаёшь медицинский экзамен. Тебе нужно дать правильный ответ на вопрос:\n{0}\n\nВарианты ответа:',
        *('%s. {%s}.' % (ALPHA_MAPPER[si], si) for si in possible_answers),
        '\nКакой из ответов {0} наиболее корректен? Обязательно ответь одной буквой.'.format('[' + ', '.join(pretty_answers) + ']'),
    ])
    input_message = prompt.format(q_dict['question'], *(q_dict[ii] for ii in possible_answers))
    memory = [SystemMessage(content=input_message)]
    for at in range(ASK_ATTEMPTS):
        ai_msg = llm.invoke(memory)
        text = ai_msg.content
        answer = parse_element(text, pretty_answers)
        # import pdb; pdb.set_trace()
        memory.append(ai_msg)
        if answer:
            break
        hc = f'Ответь одной буквой из {pretty_answers}. Если ты думаешь, что правильных ответов несколько, выбери один, самый правдоподобный'
        memory.append(HumanMessage(content=hc))
    answer = ALPHA_INV_MAPPER.get(answer)
    true_answer = q_dict.get('answer')
    moderator_msg = input_message
    moderator_answer = '{\n' + '\n###\n'.join(msg.content for msg in memory[1:]) + '\n}'
    log_answer(q_dict, possible_answers, true_answer, input_message, moderator_answer, answer)
    return answer

def get_answer_v7(llm, q_dict):
    possible_answers = extract_answer_keys(q_dict)
    pretty_answers = [ALPHA_MAPPER[an] for an in possible_answers]
    prompt = '\n'.join([
        'Ты врач, сдаёшь медицинский экзамен. Тебе нужно дать правильный ответ на вопрос:\n{0}\n\nВарианты ответа:',
        *('%s. {%s}.' % (ALPHA_MAPPER[si], si) for si in possible_answers),
        '\nКакой из ответов {0} наиболее корректен? Порассуждай последовательно про каждый из вариантов, но будь краток, в конце дай ответ'.format('[' + ', '.join(pretty_answers) + ']'),
    ])
    input_message = prompt.format(q_dict['question'], *(q_dict[ii] for ii in possible_answers))
    memory = [SystemMessage(content=input_message)]
    for at in range(ASK_ATTEMPTS):
        ai_msg = llm.invoke(memory)
        text = ai_msg.content
        answer = parse_element(text, pretty_answers)
        memory.append(ai_msg)
        if answer:
            break
        hc = f'Ответь одной буквой из {pretty_answers}. Если ты думаешь, что правильных ответов несколько, выбери один, самый правдоподобный'
        memory.append(HumanMessage(content=hc))
    answer = ALPHA_INV_MAPPER.get(answer)
    true_answer = q_dict.get('answer')
    moderator_msg = input_message
    moderator_answer = '{\n' + '\n###\n'.join(msg.content for msg in memory[1:]) + '\n}'
    log_answer(q_dict, possible_answers, true_answer, input_message, moderator_answer, answer)
    return answer

get_answer_map = {
    'v0': lambda llm, q_dict: get_answer_basic(llm, q_dict, make_input_message_v0),
    'v1': lambda llm, q_dict: get_answer_basic(llm, q_dict, make_input_message_v1),
    'v2': lambda llm, q_dict: get_answer_via_roles(llm, q_dict, make_input_message_v2),
    'v3': lambda llm, q_dict: get_answer_via_roles(llm, q_dict, make_input_message_v3),
    'v4': lambda llm, q_dict: get_answer_via_roles(llm, q_dict, make_input_message_v4),
    'v5': get_answer_v5,
    'v6': get_answer_v6,
    'v7': get_answer_v7,
}

def answer_to_test_output(q_dict, answer):
    return {'idx': q_dict['idx'], 'answer': answer}

def main(path_in, config_path='config.ini', path_out=None, answer_mode='v1'):
    run_main(
        path_in=path_in,
        config_path=config_path,
        path_out=path_out,
        get_answer_map=get_answer_map,
        answer_mode=answer_mode,
        answer_field='answer',
        answer_to_test_output=answer_to_test_output,
    )

if __name__ == '__main__':
    wrapped_fire(main)
