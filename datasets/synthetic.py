import random
import math
from copy import copy
from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from dropbox_utils import DropboxClient


class SynthDataset(ABC, Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return self._generate_datapoint(row)

    @abstractmethod
    def _generate_datapoint(self, row):
        pass

    def _verbalise_triplets(self, triplets):
        return ", ".join((self.verbalise_triplet(triplet) for triplet in triplets))

    @staticmethod
    def verbalise_triplet(triplet):
        return f"{triplet['object']} {triplet['property']} {triplet['subject']}"

    def _tokenize(self, premise, hypothesis, **kwargs):
        return self.tokenizer(premise, hypothesis, truncation=True, **kwargs)


class SynthClassificationDataset(SynthDataset):
    def __init__(
        self, data, tokenizer, negatives_rate=0.5, hypothesis_subsampling_rate=0.5
    ):
        super(SynthClassificationDataset, self).__init__(data, tokenizer)
        self.negatives_rate = negatives_rate
        self.hypothesis_subsampling_rate = hypothesis_subsampling_rate

    def _generate_datapoint(self, row):
        premise = row["premise"]
        hypothesis = row["hypothesis"]
        num_hypothesis = len(hypothesis)
        if num_hypothesis > 0 and random.random() < self.negatives_rate:
            max_rejections = math.ceil(
                num_hypothesis * self.hypothesis_subsampling_rate
            )
            num_samples = num_hypothesis - random.randint(1, max_rejections)
            subsampled_hypothesis = random.sample(hypothesis, num_samples)
            datapoint = self._tokenize(
                self._verbalise_triplets(premise),
                self._verbalise_triplets(subsampled_hypothesis),
            )
            datapoint["label"] = 0
            return datapoint
        else:
            datapoint = self._tokenize(
                self._verbalise_triplets(premise), self._verbalise_triplets(hypothesis)
            )
            datapoint["label"] = 1
            return datapoint


class SynthMLMDataset(SynthDataset):
    def __init__(self, data, tokenizer, subject_mask_rate=0.5, object_mask_rate=0.5):
        super(SynthMLMDataset, self).__init__(data, tokenizer)
        self.subject_mask_rate = subject_mask_rate
        self.object_mask_rate = object_mask_rate

    def _generate_datapoint(self, row):
        premise = row["premise"]
        hypothesis = row["hypothesis"]
        masked_hypothesis = list()
        for hypothesis_triplet in hypothesis:
            masked_triplet = copy(hypothesis_triplet)
            mask_sample = random.random()
            if mask_sample < self.subject_mask_rate:
                masked_triplet["subject"] = "<mask>"
            elif mask_sample < (self.subject_mask_rate + self.object_mask_rate):
                masked_triplet["object"] = "<mask>"
            masked_hypothesis.append(masked_triplet)
        datapoint = self._tokenize(
            self._verbalise_triplets(premise),
            self._verbalise_triplets(masked_hypothesis),
            max_length=512,
            padding="max_length",
        )
        datapoint["label"] = self._tokenize(
            self._verbalise_triplets(premise),
            self._verbalise_triplets(hypothesis),
            max_length=512,
            padding="max_length",
        )["input_ids"]
        return datapoint


def get_synth_datasets(
    data_path,
    dataset_type,
    dropbox_token,
    tokenizer,
    test_size=0.2,
    val_size=0.2,
    **dataset_kwargs,
):
    dbx = DropboxClient(dropbox_token)
    data = dbx.from_dropbox(data_path)

    train_data, test_data = train_test_split(data, test_size=test_size)
    train_data, val_data = train_test_split(train_data, test_size=val_size)

    if dataset_type == "cls":
        dataset_class = SynthClassificationDataset
    elif dataset_type == "mlm":
        dataset_class = SynthMLMDataset
    else:
        raise ValueError("Invalid dataset type")

    train_synth = dataset_class(train_data, tokenizer, **dataset_kwargs)
    val_synth = dataset_class(val_data, tokenizer, **dataset_kwargs)
    test_synth = dataset_class(test_data, tokenizer, **dataset_kwargs)
    return train_synth, val_synth, test_synth
