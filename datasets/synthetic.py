import random
import math

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from dropbox_utils import from_dropbox

class SynthDataset(Dataset):
    def __init__(self, df, tokenizer, negatives_rate=0.5, hypothesis_subsampling_rate=0.5):
        self.df = df
        self.tokenizer = tokenizer
        self.negatives_rate = negatives_rate
        self.hypothesis_subsampling_rate = hypothesis_subsampling_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self._generate_datapoint(row)

    def _generate_datapoint(self, row):
        premise = row['premise']
        hypothesis = row['hypothesis']
        num_hypothesis = len(hypothesis)
        if num_hypothesis > 0 and random.random() < self.negatives_rate:
            max_rejections = math.ceil(num_hypothesis * self.hypothesis_subsampling_rate)
            num_samples = num_hypothesis - random.randint(1, max_rejections)
            subsampled_hypothesis = random.sample(hypothesis, num_samples)
            datapoint = self._tokenize(self._verbalise_triplets(premise), self._verbalise_triplets(subsampled_hypothesis))
            datapoint['label'] = 0
            return datapoint
        else:
            datapoint = self._tokenize(self._verbalise_triplets(premise), self._verbalise_triplets(hypothesis))
            datapoint['label'] = 1
            return datapoint
        
    @staticmethod
    def _verbalise_triplets(triplets):
        return ', '.join(triplets)

    def _tokenize(self, premise, hypothesis):
        return self.tokenizer(premise, hypothesis, truncation=True)
    

def get_synth_datasets(data_path, dropbox_token, tokenizer, test_size=0.2, val_size=0.2):
    df = from_dropbox(data_path, 'synth_data.csv', dropbox_token)

    train_df, test_df = train_test_split(df, test_size=test_size)
    train_df, val_df = train_test_split(train_df, test_size=val_size)

    train_synth = SynthDataset(train_df, tokenizer)
    val_synth = SynthDataset(val_df, tokenizer)
    test_synth = SynthDataset(test_df, tokenizer)
    return train_synth, val_synth, test_synth