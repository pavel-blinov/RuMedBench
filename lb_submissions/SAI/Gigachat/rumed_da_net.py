"""Checking RumedDaNet."""

from rumed_utils import log_answer, parse_element, run_main, wrapped_fire

PROMPT = """Контекст: {context}
Вопрос: {question}

Обязательно ответь либо "Да", либо "Нет"."""

def get_answer_basic(llm, q_dict):
    possible_answers = {'да', 'нет'}
    input_message = PROMPT.format(**q_dict)
    llm_response = llm.invoke(input_message).content
    answer = parse_element(llm_response, possible_answers)
    true_answer = q_dict.get('answer')
    log_answer(q_dict, possible_answers, true_answer, input_message, llm_response, answer)
    return answer

def answer_to_test_output(q_dict, answer):
    return {'pairID': q_dict['pairID'], 'answer': answer}

get_answer_map = {
    'v0': get_answer_basic,
}

def main(path_in, config_path='config.ini', path_out=None):
    run_main(
        path_in=path_in,
        config_path=config_path,
        path_out=path_out,
        get_answer_map=get_answer_map,
        answer_mode='v0',
        answer_field='answer',
        answer_to_test_output=answer_to_test_output,
    )

if __name__ == '__main__':
    wrapped_fire(main)
