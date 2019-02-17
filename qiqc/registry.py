# Registries for preprocessing
NORMALIZER_REGISTRY = {}
TOKENIZER_REGISTRY = {}
WORD_EMBEDDING_FEATURIZER_REGISTRY = {}
WORD_EXTRA_FEATURIZER_REGISTRY = {}
SENTENCE_EXTRA_FEATURIZER_REGISTRY = {}

# Registries for training
ENCODER_REGISTRY = {}
AGGREGATOR_REGISTRY = {}
ATTENTION_REGISTRY = {}


def register_preprocessor(name):
    def register_cls(cls):
        NORMALIZER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_tokenizer(name):
    def register_cls(cls):
        TOKENIZER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_word_embedding_features(name):
    def register_cls(cls):
        WORD_EMBEDDING_FEATURIZER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_word_extra_features(name):
    def register_cls(cls):
        WORD_EXTRA_FEATURIZER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_sentence_extra_features(name):
    def register_cls(cls):
        SENTENCE_EXTRA_FEATURIZER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_encoder(name):
    def register_cls(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return register_cls


def register_aggregator(name):
    def register_cls(cls):
        AGGREGATOR_REGISTRY[name] = cls
        return cls
    return register_cls


def register_attention(name):
    def register_cls(cls):
        ATTENTION_REGISTRY[name] = cls
        return cls
    return register_cls
