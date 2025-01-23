import re
import toml
import json
import numpy as np
from tqdm import tqdm

config = toml.load("config.toml")


class BasePrompt:
    """ Base class for creating prompts """
    def __init__(self, few_shot, chain_of_thought:bool, dataset = "NYT"):
        self.few_shot = few_shot
        self.chain_of_thought = chain_of_thought
        assert dataset in ["NYT", "Cryp", "Init"]
        self.dataset = dataset
        ds = dataset.lower()
        self.query = self.load_json(config['files'][f'{ds}_semantic_test'])
        if chain_of_thought:
            self.support = self.load_json(config['files'][f'{ds}_semantic_cot'])
        else:
            self.support = self.load_json(config['files'][f'{ds}_semantic_supp'])
        self._choose_from = list(self.support.keys())

    def create_prompt(self, clue):
        self.al = set()
        support = []
        for i in range(self.few_shot):
            support_example = self._get_example() if not self.chain_of_thought\
                                                else self._get_cot_example()
            support.append(support_example)

        support.append(f"Clue: {clue['clue']} ({len(clue['answer'])})//")
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
        return f"Clue: {cl['clue']} ({len(cl['answer'])})// Answer: {cl['answer'].upper()}"

    def _get_cot_example(self):
        while True:
            key = np.random.choice(self._choose_from)
            cl = self.support[key]
            if key in self.al:
                continue
            if len(self.al) == self.few_shot:
                self.al = set()
            self.al.add(key)
            break
        return cl+"\n\n"

    @staticmethod
    def load_json(file):
        with open(file, 'r') as f:
            dataset = json.load(f)
        return dataset

    @staticmethod
    def write_json(file, dataset):
        with open(file, 'w') as f:
            json.dump(dataset, f, indent = 4)

    def create_prompts(self):
        prompts = {}
        for id, key in tqdm(self.query.items(), desc="Creating prompts"):
            prompts[id] = self.create_prompt(key)
        fname = f"./prompts/{self.dataset}_"\
                f"{'cot' if self.chain_of_thought else ''}"\
                f"_fewShot_{self.few_shot}.json"
        self.write_json(fname, prompts)

class CryptoniteSemantic(BasePrompt):
    """
    Class that implements few-shot and CoT prompt creation for the semantic
    adherence test using the Cryptonite dataset.
    """
    def __init__(self, /, few_shot:int = 0, chain_of_thought:bool = False):
        super().__init__(few_shot, chain_of_thought, dataset="Cryp")

class NYTSemantic(BasePrompt):
    """
    Class that implements few-shot and CoT prompt creation for the semantic
    adherence test using the NYT dataset.
    """
    def __init__(self, /, few_shot:int = 0, chain_of_thought:bool = False):
        super().__init__(few_shot, chain_of_thought, dataset="NYT")

class InitSemantic(BasePrompt):
    """
    Class that implements few-shot and CoT prompt creation for the semantic
    adherence test using the Init dataset.
    """
    def __init__(self, /, few_shot:int = 0, chain_of_thought:bool = False):
        super().__init__(few_shot, chain_of_thought, dataset="Init")


def main():
    NYTSemantic(few_shot = 5, chain_of_thought=False).create_prompts()
    CryptoniteSemantic(few_shot = 5, chain_of_thought=False).create_prompts()
    InitSemantic(few_shot = 5, chain_of_thought=False).create_prompts()

if __name__ == "__main__":
    main()
