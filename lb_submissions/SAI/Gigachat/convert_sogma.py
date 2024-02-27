import json
import sys
from pathlib import Path

import fire
import xmltodict

def read_xml_sogma_tasks(xml_content):
    # https://geetest.ru/tests/terapiya_(dlya_internov)_sogma_/download
    tree = xmltodict.parse(xml_content)
    questions = tree['geetest']['test']['questions']['question']
    for qu in questions:
        task = dict(id=qu['@id'], question=qu['text'])
        for an in qu['answers']['answer']:
            task[an['@num']] = an['#text']
        answers = [an['@num'] for an in qu['answers']['answer'] if an['@isCorrect'] == '1']
        if len(answers) != 1:
            continue
        task['answer'] = answers[0]
        yield task

def main(path_in='sogma-test.xml', path_out='RuMedTest--sogma--dev.jsonl'):
    assert path_in.endswith('.xml')
    assert path_out.endswith('.jsonl')

    xml_content = Path(path_in).read_text()
    tasks = list(read_xml_sogma_tasks(xml_content))
    jsonl_content = '\n'.join(json.dumps(task, ensure_ascii=False) for task in tasks)
    Path(path_out).write_text(jsonl_content)
    print('Done! Converted {0} tasks!'.format(len(tasks)))

if __name__ == '__main__':
    fire.Fire(main)
