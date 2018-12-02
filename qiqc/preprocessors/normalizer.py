import json
from pathlib import Path


class SentenceNormalizationPipeline(object):

    def __init__(self, *normalizers):
        self.normalizers = normalizers

    def __call__(self, sentence):
        tokens = []
        for token in sentence.split():
            for normalizer in self.normalizers:
                token = normalizer(token)
            tokens.append(token)
        return ' '.join(tokens)


class RulebasedNormalizer(object):

    def __init__(self, rule):
        assert isinstance(rule, dict)
        self.rule = rule

    def __call__(self, x):
        return self.rule.get(x, x)


class ContractionNormalizer(RulebasedNormalizer):

    def __init__(self):
        self.filepath = Path(__file__).parent / 'rules/contraction.json'
        rule = json.load(open(self.filepath))
        super().__init__(rule)


class PunktNormalizer(RulebasedNormalizer):

    def __init__(self):
        self.filepath = Path(__file__).parent / 'rules/punkt.json'
        rule = json.load(open(self.filepath))
        super().__init__(rule)


class TypoNormalizer(RulebasedNormalizer):

    def __init__(self):
        self.filepath = Path(__file__).parent / 'rules/typo.json'
        rule = json.load(open(self.filepath))
        super().__init__(rule)
