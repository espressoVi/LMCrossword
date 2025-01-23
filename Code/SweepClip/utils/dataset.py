import json
from tqdm import tqdm
from utils.primitives import Crossword, config

class Dataset:
    def __init__(self):
        _data = self.load_data()
        self.data = self.parse_data(_data)

    @staticmethod
    def parse_data(dataset):
        results = []
        for data in tqdm(dataset, desc = "Parsing dataset"):
            c = Crossword(data['number'], data['clues'], data['board'])
            results.append(c)
        return results

    @staticmethod
    def load_data():
        with open(config['files']['test'], 'r') as f:
            dataset = json.load(f)
        return dataset['data']

    def __iter__(self):
        self.counter = 0
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.counter >= len(self):
            raise StopIteration
        res = self.data[self.counter]
        self.counter += 1
        return res

    def __getitem__(self, idx):
        if idx < len(self):
            return self.data[idx]
        raise IndexError(f"{idx} is out of range for dataset of size {len(self)}")
