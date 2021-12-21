def sick_tokenize_function(example, tokenizer):
    return tokenizer(example["sentence_A"], example["sentence_B"], truncation=True)

def get_sick_dataset(tokenizer):
    raw_datasets = load_dataset('sick')
    tokenized_datasets = raw_datasets.map(sick_tokenize_function, batched=True)
    return tokenized_datasets

class SickWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def get_tokenized_dataset(self):
        raw_datasets = load_dataset('sick')
        tokenized_datasets = raw_datasets.map(self._tokenize, batched=True)
        return tokenized_datasets

    def _tokenize(self, example):
        return self.tokenizer(example["sentence_A"], example["sentence_B"], truncation=True)