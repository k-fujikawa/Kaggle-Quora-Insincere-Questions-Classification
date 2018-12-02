import nltk
import torch
from torch import nn

import qiqc.builder as QB
import qiqc.featurizers as QF
import qiqc.models as QM
import qiqc.preprocessors as QP


def build_preprocessor(config):
    pipeline = QP.PreprocessPipeline(
        QP.SentenceNormalizationPipeline(
            QP.TypoNormalizer(),
        ),
        nltk.word_tokenize,
    )
    return pipeline


def build_featurizer(config, tokens):
    pretrained_vector = QF.load_pretrained_vector(
        config['featurizer']['pretrain'], test=config['test'])
    w2v = QM.Word2VecEx(**config['featurizer']['train'])
    featurizer = QF.Word2VecFeaturizer(
        model=w2v,
        maxlen=config['featurizer']['maxlen'],
        standardize=config['featurizer']['standardize'],
    )
    featurizer.model.build_vocab_with_pretraining(
        tokens, pretrained_vector, keep_raw_vocab=True)
    for i in range(config['featurizer']['n_finetune']):
        featurizer.model.reset_pretrained_vector()
        featurizer.model.train(
            tokens, total_examples=len(tokens), epochs=1)
    featurizer.build_w2vtable()
    return featurizer


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed = QM.WordEmbedding(
            *config['embedding_matrix'].shape,
            n_hidden=config['embed']['n_hidden'],
            freeze_embed=config['embed']['freeze_embed'],
            pretrained_vectors=config['embedding_matrix'],
            position=config['embed']['position'],
            hidden_bn=config['embed']['hidden_bn'],
        )
        self.encoder = nn.GRU(
            input_size=self.embed.out_dim,
            hidden_size=config['encoder']['n_hidden'],
            num_layers=config['encoder']['n_layers'],
            dropout=config['encoder']['dropout'],
            bidirectional=True,
            batch_first=True,
        )
        self.aggregator = QB.build_aggregator(
            config['encoder']['aggregator'],
        )

    def forward(self, X, mask):
        h = self.embed(X)
        h, _ = self.encoder(h)
        h = self.aggregator(h, mask)
        return h


def build_model(config):
    encoder = Encoder(config['model'])
    clf = QM.BinaryClassifier(config['model'], encoder)
    return clf


def build_optimizer(config, model):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config['optimizer']['lr']))
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=float(config['optimizer']['lr']))
    return optimizer
