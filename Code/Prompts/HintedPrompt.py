import re
import toml
import json
import numpy as np
from tqdm import tqdm
from Prompt import BasePrompt
import numpy as np

config = toml.load("config.toml")


class HintedPrompt(BasePrompt):
    """ Base class for creating hinted prompts """
    def __init__(self, hint = 0.5, few_shot = 5, dataset = "NYT"):
        assert 0 < hint < 1
        self.hint = hint
        super().__init__(few_shot, False, dataset)

    def create_prompt(self, clue):
        self.al = set()
        support = []
        for i in range(self.few_shot):
            support_example = self._get_example() if not self.chain_of_thought\
                                                else self._get_cot_example()
            support.append(support_example)

        support.append(self._hinter(clue))
        support[-1] += " Answer: " if not self.chain_of_thought\
                                    else config['prompt']['cot']
        prompt = [
                {
                    'role':'system', 
                    'content':config['prompt']['system'] \
                            if not self.chain_of_thought \
                            else config['prompt']['cot_system'],
                },
                {'role':'user', 'content':"\n".join(support)},
        ]
        return prompt

    def _get_example(self):
        while True:
            key = np.random.choice(self._choose_from)
            cl = self.support[key]
            if cl['answer'] in self.al:
                continue
            if len(self.al) == self.few_shot:
                self.al = set()
            self.al.add(cl['answer'])
            break
        return self._hinter(cl) + f" Answer: {cl['answer'].upper()}"

    def _hinter(self, cl):
        ans = list(cl['answer'].upper())
        how_many = int(np.round(len(ans)*self.hint))
        mask = set(np.random.choice(len(ans), size = how_many, replace = False))
        ans = [i if idx in mask else "_" for idx, i in enumerate(ans)]
        ans = " ".join(ans)
        return f"Clue: {cl['clue']} ({len(cl['answer'])})// {ans} =>"

    def create_prompts(self):
        prompts = {}
        for id, key in tqdm(self.query.items(), desc="Creating prompts"):
            prompts[id] = self.create_prompt(key)
        hnum = int(1/self.hint)
        fname = f"./prompts/{self.dataset}_"\
                f"{'cot' if self.chain_of_thought else ''}"\
                f"_Hinted_{hnum}_fewShot_{self.few_shot}.json"
        self.write_json(fname, prompts)
 

class NYTHinted(HintedPrompt):
    """
    Class that implements few-shot and CoT prompt creation for the semantic
    adherence test using the NYT dataset.
    """
    def __init__(self, /, hint = 0.5, few_shot:int = 0, chain_of_thought:bool = False):
        super().__init__(hint, few_shot, chain_of_thought, dataset="NYT")

class InitHinted(HintedPrompt):
    """
    Class that implements few-shot and CoT prompt creation for the semantic
    adherence test using the Init dataset.
    """
    def __init__(self, /, hint = 0.5, few_shot:int = 0, chain_of_thought:bool = False):
        super().__init__(hint, few_shot, chain_of_thought, dataset="Init")


def main():
    NYTHinted(hint = 0.25, few_shot = 5).create_prompts()
    NYTHinted(hint = 0.5, few_shot = 5).create_prompts()
    InitHinted(hint = 0.25, few_shot = 5).create_prompts()
    InitHinted(hint = 0.5, few_shot = 5).create_prompts()
 
if __name__ == "__main__":
    main()
