import json
import random


class RuletakerDataset:
    def __init__(self, filepath: str, tokenizer):
        self.data = self._load_data(filepath)
        self.tokenizer = tokenizer

    @staticmethod
    def _load_data(filepath: str):
        data = list()
        with open(filepath) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        world = self.data[idx]
        premise = world["context"]
        hypothesis_version = random.choice(world["questions"])
        hypothesis = hypothesis_version["text"]
        label = int(hypothesis_version["label"])
        datapoint = self._tokenize(premise, hypothesis)
        datapoint["label"] = label
        return datapoint

    def _tokenize(self, premise, hypothesis):
        return self.tokenizer(premise, hypothesis, truncation=True)
