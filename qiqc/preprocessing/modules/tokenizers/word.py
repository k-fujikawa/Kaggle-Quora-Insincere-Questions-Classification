import nltk

from qiqc.registry import register_tokenizer
from _qiqc.preprocessing.modules.tokenizers.word import cysplit


register_tokenizer('space')(cysplit)
register_tokenizer('word_tokenize')(nltk.word_tokenize)
