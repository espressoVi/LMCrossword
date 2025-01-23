import re
import toml
import json
import numpy as np
from tqdm import tqdm
from Prompt import BasePrompt

config = toml.load("config.toml")


class CountingPrompt(BasePrompt):
    """ Base class for creating counting prompts """
    def __init__(self, few_shot):
        super().__init__(few_shot, False)
        self.query = self.load_json(config['files'][f'counting_test'])
        self.support = self.load_json(config['files'][f'counting_supp'])
        self._choose_from = list(self.support.keys())

    def create_prompt(self, clue):
        self.al = set()
        support = []
        for i in range(self.few_shot):
            support_example = self._get_example()
            support.append(support_example)

        support.append(f"Word: {clue} // ")
        prompt = [
                {'role':'system', 'content':config['prompt']['counting']},
                {'role':'user', 'content':"\n".join(support)},
        ]
        return prompt

    def _get_example(self):
        while True:
            key = np.random.choice(self._choose_from)
            cl = self.support[key]
            if cl in self.al:
                continue
            if len(self.al) == self.few_shot:
                self.al = set()
            self.al.add(cl)
            break
        return f"Word: {cl} // {len(cl)}"

    def create_prompts(self):
        prompts = {}
        for key, word in tqdm(self.query.items(), desc="Creating prompts"):
            prompts[key] = self.create_prompt(word)
        fname = f"./prompts/counting_fewShot_{self.few_shot}.json"
        self.write_json(fname, prompts)


def main():
    CountingPrompt(5).create_prompts()
    
if __name__ == "__main__":
    main()
