import torch
from tokenizers.models import WordLevel
# from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from datasets import load_dataset

from pathlib import Path



def get_all_sentences(data_set, lang):
    '''
      data_set is a HuggingFace dataset object, like the one returned by load_dataset()
      It returns a generator that yields all sentences in the specified language
      lang is a string, like 'en' or 'it' or any other language code
      Example usage:
      for sentence in get_all_sentences(data_set, 'en'):
          print(sentence)

      data_set is a list of dictionaries, each expected to have a 'translation' field, which is a dictionary
      with language codes as keys (like 'en', 'de', etc.) and the corresponding sentences as values:
      [
          {'id': '2', 'translation': {'en': 'Charlotte Bronte', 'it': 'Charlotte Brontë'}}
          {'id': '3', 'translation': {'en': 'CHAPTER I', 'it': 'PARTE PRIMA'}}
      ]

    '''
    for item in data_set:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, data_set, lang):
    # config['tokenizer_file'] = "tokenizer_{lang}.json"
    # given the language, return the path to the tokenizer file like "tokenizer_en.json", "tokenizer_de.json", etc.
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        # for a word to appear in the vocabulary, it must appear at least `min_frequency` times
        # in the training data. This is useful to avoid having too many rare words in the vocabulary.
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)

        # tokenizer.train(files, trainer)
        tokenizer.train_from_iterator(
            get_all_sentences(data_set, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
