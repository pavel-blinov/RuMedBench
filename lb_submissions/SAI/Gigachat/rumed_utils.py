import configparser
import json
import logging
import time
from datetime import datetime as dt
from logging.config import dictConfig
from pathlib import Path

import fire
import httpx
from gigachat.exceptions import GigaChatException
from langchain.cache import SQLiteCache
from langchain.chat_models.gigachat import GigaChat
from langchain.globals import set_llm_cache
from tqdm import tqdm

def extract_tags(path_in: Path):
    tags = []
    if path_in.parent.name.startswith('Ru'):
        tags.append(path_in.parent.name)
    tags.append(path_in.stem)
    return tags

def init_logging(path_in, answer_mode):
    path_in = Path(path_in)
    Path('./logs').mkdir(exist_ok=True)
    dt_now_pretty = dt.strftime(dt.now(), '%Y-%m-%d--%H-%M-%S')
    tags = [dt_now_pretty] + extract_tags(path_in) + [answer_mode]
    filename_log = './logs/{0}.log'.format('--'.join(tags))

    logging_config = {
        'version': 1,
        'handlers': {
            'file_handler': {
                'class': 'logging.FileHandler',
                'filename': filename_log,
                'level': 'DEBUG',
                'formatter': 'standard',
            },
            'benchmarks_file_handler': {
                'class': 'logging.FileHandler',
                'filename': 'benchmarks.log',
                'level': 'INFO',
                'formatter': 'standard',
            },
            'stream_handler': {
                'class': 'logging.StreamHandler',
                'level': 'WARNING',
                'formatter': 'standard',
            },
        },
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(message)s',
                'datefmt': '%Y-%m-%d--%H-%M-%S',
            },
        },
        'loggers': {
            'root': {
                'level': 'DEBUG',
                'handlers': ['file_handler', 'stream_handler'],
            },
            'benchmarks': {
                'level': 'INFO',
                'handlers': ['benchmarks_file_handler'],
            }
        }
    }
    dictConfig(logging_config)
    logging.warning('Logs_path: %s', filename_log)

GIGACHAT_MODEL = 'GigaChat-Pro'

def create_llm_gigachat(config):
    credentials = dict(config['credentials'])
    base_url = config['base_url']['base_url']
    logging.info('credentials: %s', credentials)
    logging.info('base_url: %s', base_url)

    # https://python.langchain.com/docs/modules/model_io/llms/llm_caching
    user = config['credentials']['user']
    database_path = "{0}.langchain.db".format(user)
    set_llm_cache(SQLiteCache(database_path=database_path))

    return GigaChat(
        verify_ssl_certs=False,
        profanity_check=False,
        model=GIGACHAT_MODEL,
        base_url=base_url,
        **credentials,
    )

def parse_element(answer, elements):
    fst_word = answer.split()[0].strip(',.').strip().lower()
    if fst_word in elements:
        return fst_word
    return None

def format_accuracy(correct, total):
    return '{0:.2f} %'.format(correct / total * 100)

ATTEMPTS = 10
WAIT_SECONDS = 6
def repeater(callback, skip_ex):
    def wrapped_callback(*args, **kwargs):
        wait_s = WAIT_SECONDS
        for at in range(1, ATTEMPTS + 1):
            try:
                return callback(*args, **kwargs)
            except Exception as ex:
                if skip_ex and isinstance(ex, skip_ex):
                    logging.warning('Failed to execute: %s, attempt=%d', ex, at)
                    time.sleep(wait_s)
                    wait_s *= 2
                    if at == ATTEMPTS:
                        logging.exception('Attempts out...')
                        raise ex
                else:
                    raise ex
    return wrapped_callback

def init_llm(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return create_llm_gigachat(config)
    
def read_json_tasks(path_in):
    tasks = [json.loads(line) for line in path_in.read_text().strip().splitlines()]
    return tasks

CONNECTION_EXCEPTIONS = (GigaChatException, httpx.HTTPError, json.decoder.JSONDecodeError)

def benchmark_check(path_in, llm, tasks, get_answer, answer_field, tags=None):
    logging.warning('Path_in: %s', path_in)
    logging.warning('Tasks: %d', len(tasks))
    correct_total = 0
    pbar = tqdm(range(len(tasks)))
    w_get_answer = repeater(get_answer, skip_ex=CONNECTION_EXCEPTIONS)
    for ti in pbar:
        td = tasks[ti]
        answer = w_get_answer(llm, td)
        true_answer = td[answer_field]
        check = answer == true_answer
        correct_total += check
        acc = format_accuracy(correct_total, ti + 1)
        pbar.set_description('acc: {0}'.format(acc))
        if ti % 10 == 0:
            logging.info('index=%d, acc: %s', ti, acc)
    tags = extract_tags(path_in) + (tags or [])
    b_info = 'Tags: {0}, final accuracy: {1}'.format(' '.join(tags), format_accuracy(correct_total, len(tasks)))
    logging.getLogger('benchmarks').info(b_info)
    logging.warning('Done! %s', b_info)

def benchmark_test(llm, tasks, path_out, get_answer, answer_to_test_output):
    logging.warning('Tasks: %d', len(tasks))
    lines = []
    pbar = tqdm(range(len(tasks)))
    w_get_answer = repeater(get_answer, skip_ex=CONNECTION_EXCEPTIONS)
    for ti in pbar:
        td = tasks[ti]
        answer = w_get_answer(llm, td)
        answer_output = answer_to_test_output(td, answer)
        lines.append(json.dumps(answer_output, ensure_ascii=False))
    Path(path_out).write_text('\n'.join(lines))
    logging.info('Done! Saved to %s', path_out)

def log_answer(q_dict, possible_answers, true_answer, input_message, llm_response, answer):
    log_callback = logging.debug if (answer is not None) else logging.warning
    # todo fix, use `None` here
    if true_answer is not None:
        check = answer == true_answer
        log_callback('input_message: %s\nllm_response: %s\nanswer: %s\ntrue_answer: %s\ncorrect: %s', input_message, llm_response, answer, true_answer, check)
    else:
        log_callback('input_message: %s\nllm_response: %s\nanswer: %s', input_message, llm_response, answer)

    if answer not in possible_answers:
        logging.warning('Expected answer `{0}` not in possible answers `[{1}]`, q_dict: {2}'.format(answer, ', '.join(possible_answers), q_dict))

def choose_get_answer(get_answer_map, answer_mode):
    get_answer = get_answer_map.get(answer_mode)
    if get_answer is None:
        raise ValueError('Supported answer versions: {0}, found: {1}'.format(list(get_answer_map.keys()), answer_mode))
    return get_answer

def run_main(path_in, config_path, path_out, get_answer_map, answer_mode, answer_field, answer_to_test_output):
    get_answer = choose_get_answer(get_answer_map, answer_mode)
    path_in = Path(path_in)
    if not path_in.exists():
        raise ValueError('`path_in`=`{0}` not exists!'.format(path_in))
    init_logging(path_in, answer_mode)
    logging.warning('Answer mode: {0}'.format(answer_mode))
    tags = [answer_mode]

    llm = init_llm(config_path)
    tasks = read_json_tasks(path_in)
    if any(sub in path_in.stem for sub in ('dev', 'train')):
        benchmark_check(path_in, llm, tasks, get_answer, answer_field, tags=tags)
    elif 'test' in path_in.stem:
        if path_out is None:
            raise ValueError('`path_out` should be passed')
        benchmark_test(llm, tasks, path_out, get_answer, answer_to_test_output)
    else:
        raise ValueError('Can not recognize mode, expected `dev`, `train` or `test` in `path_in`')

def wrapped_fire(main):
    try:
        fire.Fire(main)
    except KeyboardInterrupt:
        logging.warning('Cancelled!')
        raise KeyboardInterrupt('Cancelled!')

