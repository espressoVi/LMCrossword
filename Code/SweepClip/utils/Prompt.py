import toml
import json
import numpy as np
from tqdm import tqdm
import numpy as np

config = toml.load("config.toml")


class Prompter:
    """ Class for creating prompts corresponding to queries."""
    def __init__(self, few_shot):
        self.few_shot = few_shot
        self.support = self.load_json(config['files']['support'])
        self._choose_from = list(self.support.keys())

    def create_prompt(self, clue:str, hint:str = None) -> list[dict]:
        """
        Method that creates the prompt to be passed to an LLM. The prom

        Args:
            str : The clue being queried.
            Optional[str] : character hint (if any).

        Returns:
            prompt (list[dict]) : The prompt to be passed to the LLM.

        Formats:
            clue = 'Clue: <clue text> (<answer length>)'
            hint = '_ _ X _ Y'
        """
        support = []
        length = int(clue[clue.rfind("(")+1:clue.rfind(")")])
        for i in range(self.few_shot):
            support_example = self._get_example(length) if hint is None\
                    else self._get_hinted_example(hint)
            support.append(support_example)

        if hint is None:
            query = f"{clue}// Answer: "
        else:
            query = f"{clue}// {hint} => Answer: "
        support.append(query)
        prompt = [
                    {'role':'system', 'content':config['prompts']['system']},
                    {'role':'user', 'content':"\n".join(support)},
        ]
        return prompt

    def _get_example(self, length = None):
        if length is None:
            cl = self.support[np.random.choice(self._choose_from)]
            return f"{cl['clue']}// Answer: {cl['answer'].upper()}"
        else:
            while True:
                cl = self.support[np.random.choice(self._choose_from)]
                if len(cl['answer']) == length:
                    return f"{cl['clue']}// Answer: {cl['answer'].upper()}"

    def _get_hinted_example(self, hint):
        hh = len([i for i in hint if i.isalpha() or i == config['sep']])
        hint_percent = 1 - hint.count(config['sep'])/hh
        cl = self.support[np.random.choice(self._choose_from)]
        answer = cl['answer']
        ans = list(answer.upper())
        how_many = max(1, int(np.round(len(ans)*hint_percent)))
        mask = set(np.random.choice(len(ans), size = how_many, replace = False))
        ans = " ".join([i if idx in mask else "_" for idx, i in enumerate(ans)])
        return f"{cl['clue']}// {ans} => Answer: {cl['answer'].upper()}"

    @staticmethod
    def load_json(file):
        with open(file, 'r') as f:
            dataset = json.load(f)
        return dataset
