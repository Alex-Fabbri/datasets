import tensorflow_datasets as tfds

import torch
from torchtext.data import Field, BucketIterator
import torchtext


class TFDSTorchTextDataset(torchtext.data.Dataset):
    """Example for loading summarization data -
       follows https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py
    """
    def __init__(self, datasetid, src_key, tgt_key, split, fields):
        """
        Args:
            datasetid (string): identifier for tfds.load
            split (string): train/val/test to load
            fields (dict): string : torchtext field
        """
        summ_dataset, info = tfds.load(datasetid, split=split, with_info=True)
        self.summ_dataset = summ_dataset
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.info = info
        self.datasplit = split
        self.fields = fields
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]
        # probably can be optimized somewhere
        self.examples = []
        for ex in self.summ_dataset:
            docs = ex[self.src_key].numpy().decode("utf-8")
            summ = ex[self.tgt_key].numpy().decode("utf-8")
            self.examples.append(torchtext.data.Example.fromlist([docs, summ], list(self.fields.items())))

    def __len__(self):
        return self.info.splits[self.datasplit].num_examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __iter__(self):
        for ex in self.examples:
            yield ex

    def __getattr__(self, attr):
        if attr in self.fields:
            for ex in self.examples:
                yield getattr(ex, attr)

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs):
        raise NotImplementedError("This functionality hasn't yet been added.")

    def split(self, split_ratio=0.7, stratified=False, strata_field='label',
              random_state=None):
        raise NotImplementedError("To remove HTML markup, use BeautifulSoup's get_text() function")

if __name__ == "__main__":
    # building off of the torchtext tutorial here: https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    DOCUMENT = Field(tokenize="spacy",
                     tokenizer_language="en",
                     init_token='<sos>',
                     eos_token='<eos>',
                     lower=True, 
                     fix_length=400)
    SUMMARY = Field(tokenize="spacy",
                    tokenizer_language="en",
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    fix_length=100)

    # train_data = TFDSTorchTextDataset('multi_news', 'document', 'summary', 'train', {"document": DOCUMENT, "summary": SUMMARY})
    # valid_data = TFDSTorchTextDataset('multi_news', 'document', 'summary', 'validation', {"document": DOCUMENT, "summary": SUMMARY})
    # test_data = TFDSTorchTextDataset('multi_news', 'document', 'summary', 'test', {"document": DOCUMENT, "summary": SUMMARY})
    train_data = TFDSTorchTextDataset('cnn_dailymail', 'article', 'highlights', 'train', {"article": DOCUMENT, "highlights": SUMMARY})
    valid_data = TFDSTorchTextDataset('cnn_dailymail', 'article', 'highlights', 'validation', {"article": DOCUMENT, "highlights": SUMMARY})
    test_data = TFDSTorchTextDataset('cnn_dailymail', 'article', 'highlights', 'test', {"article": DOCUMENT, "highlights": SUMMARY})
    SUMMARY.build_vocab(train_data, min_freq = 2)
    DOCUMENT.build_vocab(train_data, min_freq = 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    train_iterator, valid_data, test_iterator=BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)

    for _, batch in enumerate(train_iterator):
        src = batch.article
        tgt = batch.highlights
