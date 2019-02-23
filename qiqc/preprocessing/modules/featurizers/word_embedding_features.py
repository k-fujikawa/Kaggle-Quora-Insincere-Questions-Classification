import numpy as np
from gensim.models import Word2Vec, FastText

from qiqc.registry import register_word_embedding_features


@register_word_embedding_features('pretrained')
class PretrainedVectorFeaturizer(object):

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

    @classmethod
    def add_args(self, parser):
        pass

    def __call__(self, features, datasets):
        # Nothing to do
        return features


class Any2VecFeaturizer(object):

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

    def build_fillvalue(self, mode, initialW):
        n_embed = initialW.shape[1]
        n_fill = initialW[self.vocab.unk].shape
        assert mode in {'zeros', 'mean', 'noise'}
        if mode == 'zeros':
            return np.zeros(n_embed, 'f')
        elif mode == 'mean':
            return initialW.mean(axis=0)
        elif mode == 'noise':
            mean, std = initialW.mean(), initialW.std()
            return np.random.normal(mean, std, (n_fill, n_embed))

    def __call__(self, features, datasets):
        tokens = np.concatenate([d.tokens for d in datasets])
        model = self.build_model()
        model.build_vocab_from_freq(self.vocab.word_freq)
        initialW = features.copy()
        initialW[self.vocab.unk] = self.build_fillvalue(
            self.config.finetune_word2vec_init_unk, initialW)
        idxmap = np.array(
            [self.vocab.token2id[w] for w in model.wv.index2entity])
        model = self.initialize(model, initialW, idxmap)
        model.train(tokens, total_examples=len(tokens), epochs=model.epochs)
        finetunedW = np.zeros((initialW.shape), 'f')
        for i, word in enumerate(self.vocab.token2id):
            if word in model.wv:
                finetunedW[i] = model.wv.get_vector(word)
        return finetunedW


@register_word_embedding_features('word2vec')
class Word2VecFeaturizer(Any2VecFeaturizer):

    @classmethod
    def add_args(self, parser):
        parser.add_argument('--finetune-word2vec-init-unk', type=str,
                            choices=['zeros', 'mean', 'noise'])
        parser.add_argument('--finetune-word2vec-mincount', type=int)
        parser.add_argument('--finetune-word2vec-workers', type=int)
        parser.add_argument('--finetune-word2vec-iter', type=int)
        parser.add_argument('--finetune-word2vec-size', type=int)
        parser.add_argument('--finetune-word2vec-window', type=int, default=5)
        parser.add_argument('--finetune-word2vec-sorted-vocab', type=int,
                            default=0)
        parser.add_argument('--finetune-word2vec-sg', type=int, choices=[0, 1])

    def build_model(self):
        model = Word2Vec(
            min_count=self.config.finetune_word2vec_mincount,
            workers=self.config.finetune_word2vec_workers,
            iter=self.config.finetune_word2vec_iter,
            size=self.config.finetune_word2vec_size,
            window=self.config.finetune_word2vec_window,
            sg=self.config.finetune_word2vec_sg,
        )
        return model

    def initialize(self, model, initialW, idxmap):
        model.wv.vectors[:] = initialW[idxmap]
        model.trainables.syn1neg[:] = initialW[idxmap]
        return model


@register_word_embedding_features('fasttext')
class FastTextFeaturizer(Any2VecFeaturizer):

    @classmethod
    def add_args(self, parser):
        parser.add_argument('--finetune-fasttext-init-unk', type=str,
                            choices=['zeros', 'mean', 'noise'])
        parser.add_argument('--finetune-fasttext-mincount', type=int)
        parser.add_argument('--finetune-fasttext-workers', type=int)
        parser.add_argument('--finetune-fasttext-iter', type=int)
        parser.add_argument('--finetune-fasttext-size', type=int)
        parser.add_argument('--finetune-fasttext-sg', type=int, choices=[0, 1])
        parser.add_argument('--finetune-fasttext-min_n', type=int)
        parser.add_argument('--finetune-fasttext-max_n', type=int)

    def build_model(self):
        model = FastText(
            min_count=self.config.finetune_fasttext_mincount,
            workers=self.config.finetune_fasttext_workers,
            iter=self.config.finetune_fasttext_iter,
            size=self.config.finetune_fasttext_size,
            sg=self.config.finetune_fasttext_sg,
            min_n=self.config.finetune_fasttext_min_n,
            max_n=self.config.finetune_fasttext_max_n,
        )
        return model

    def initialize(self, model, initialW, idxmap):
        model.wv.vectors[:] = initialW[idxmap]
        model.wv.vectors_vocab[:] = initialW[idxmap]
        model.trainables.syn1neg[:] = initialW[idxmap]
        return model
