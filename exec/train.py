import argparse
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import qiqc
from qiqc.datasets import load_qiqc, build_datasets
from qiqc.preprocessing.modules import load_pretrained_vectors
from qiqc.training import classification_metrics, ClassificationResult
from qiqc.utils import set_seed, load_module


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfile', '-m', type=Path, required=True)
    _args, others = parser.parse_known_args(args)

    modules = load_module(_args.modelfile)
    config = modules.ExperimentConfigBuilder().build(args=args)
    qiqc.utils.rmtree_after_confirmation(config.outdir, config.test)
    train(config, modules)


def train(config, modules):
    print(config)
    start = time.time()
    set_seed(config.seed)
    config.outdir.mkdir(parents=True, exist_ok=True)

    build_model = modules.build_model
    Preprocessor = modules.Preprocessor
    TextNormalizer = modules.TextNormalizer
    TextTokenizer = modules.TextTokenizer
    WordEmbeddingFeaturizer = modules.WordEmbeddingFeaturizer
    WordExtraFeaturizer = modules.WordExtraFeaturizer
    SentenceExtraFeaturizer = modules.SentenceExtraFeaturizer
    Ensembler = modules.Ensembler

    train_df, submit_df = load_qiqc(n_rows=config.n_rows)
    datasets = build_datasets(train_df, submit_df, config.holdout, config.seed)
    train_dataset, test_dataset, submit_dataset = datasets

    print('Tokenize texts...')
    preprocessor = Preprocessor()
    normalizer = TextNormalizer(config)
    tokenizer = TextTokenizer(config)
    train_dataset.tokens, test_dataset.tokens, submit_dataset.tokens = \
        preprocessor.tokenize(datasets, normalizer, tokenizer)

    print('Build vocabulary...')
    vocab = preprocessor.build_vocab(datasets, config)

    print('Build token ids...')
    train_dataset.tids, test_dataset.tids, submit_dataset.tids = \
        preprocessor.build_tokenids(datasets, vocab, config)

    print('Build sentence extra features...')
    sentence_extra_featurizer = SentenceExtraFeaturizer(config)
    train_dataset._X2, test_dataset._X2, submit_dataset._X2 = \
        preprocessor.build_sentence_features(
            datasets, sentence_extra_featurizer)
    [d.build(config.device) for d in datasets]

    print('Load pretrained vectors...')
    pretrained_vectors = load_pretrained_vectors(
        config.use_pretrained_vectors, vocab.token2id, test=config.test)

    print('Build word embedding matrix...')
    word_embedding_featurizer = WordEmbeddingFeaturizer(config, vocab)
    embedding_matrices = preprocessor.build_embedding_matrices(
        datasets, word_embedding_featurizer, vocab, pretrained_vectors)

    print('Build word extra features...')
    word_extra_featurizer = WordExtraFeaturizer(config, vocab)
    word_extra_features = word_extra_featurizer(vocab)

    print('Build models...')
    word_features_cv = [
        preprocessor.build_word_features(
            word_embedding_featurizer, embedding_matrices, word_extra_features)
        for i in range(config.cv)]

    models = [
        build_model(
            config, word_features, sentence_extra_featurizer.n_dims
        ) for word_features in word_features_cv]

    print('Start training...')
    splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=config.cv, shuffle=True, random_state=config.seed)
    train_results, valid_results = [], []
    best_models = []

    for i_cv, (train_indices, valid_indices) in enumerate(
            splitter.split(train_dataset.df, train_dataset.df.target)):
        if config.cv_part is not None and i_cv >= config.cv_part:
            break
        train_tensor = train_dataset.build_labeled_dataset(train_indices)
        valid_tensor = train_dataset.build_labeled_dataset(valid_indices)
        valid_iter = DataLoader(
            valid_tensor, batch_size=config.batchsize_valid)

        model = models.pop(0)
        model = model.to_device(config.device)
        model_snapshots = []
        optimizer = torch.optim.Adam(model.parameters(), config.lr)
        train_result = ClassificationResult('train', config.outdir, str(i_cv))
        valid_result = ClassificationResult('valid', config.outdir, str(i_cv))

        batchsize = config.batchsize
        for epoch in range(config.epochs):
            if epoch in config.scale_batchsize:
                batchsize *= 2
                print(f'Batchsize: {batchsize}')
            epoch_start = time.time()
            sampler = None
            train_iter = DataLoader(
                train_tensor, sampler=sampler, drop_last=True,
                batch_size=batchsize, shuffle=sampler is None)
            _summary = []

            # Training loop
            for i, batch in enumerate(
                    tqdm(train_iter, desc='train', leave=False)):
                model.train()
                optimizer.zero_grad()
                loss, output = model.calc_loss(*batch)
                loss.backward()
                optimizer.step()
                train_result.add_record(**output)
            train_result.calc_score(epoch)
            _summary.append(train_result.summary.iloc[-1])

            # Validation loop
            if epoch >= config.validate_from:
                for i, batch in enumerate(
                        tqdm(valid_iter, desc='valid', leave=False)):
                    model.eval()
                    loss, output = model.calc_loss(*batch)
                    valid_result.add_record(**output)
                valid_result.calc_score(epoch)
                _summary.append(valid_result.summary.iloc[-1])

                _model = deepcopy(model)
                _model.threshold = valid_result.summary.threshold[epoch]
                model_snapshots.append(_model)

            summary = pd.DataFrame(_summary).set_index('name')
            epoch_time = time.time() - epoch_start
            pbar = '#' * (i_cv + 1) + '-' * (config.cv - 1 - i_cv)
            tqdm.write(f'\n{pbar} cv: {i_cv} / {config.cv}, epoch {epoch}, '
                       f'time: {epoch_time}')
            tqdm.write(str(summary))

        train_results.append(train_result)
        valid_results.append(valid_result)
        best_indices = valid_result.summary.fbeta.argsort()[::-1]
        best_models.extend([model_snapshots[i] for i in
                            best_indices[:config.ensembler_n_snapshots]])

    # Build ensembler
    train_X, train_X2, train_t = \
        train_dataset.X, train_dataset.X2, train_dataset.t
    ensembler = Ensembler(config, best_models, valid_results)
    ensembler.fit(train_X, train_X2, train_t)
    scores = dict(
        valid_fbeta=np.array([r.best_fbeta for r in valid_results]).mean(),
        valid_epoch=np.array([r.best_epoch for r in valid_results]).mean(),
        threshold_cv=ensembler.threshold_cv,
        threshold=ensembler.threshold,
        elapsed_time=time.time() - start,
    )

    if config.holdout:
        test_X, test_X2, test_t = \
            test_dataset.X, test_dataset.X2, test_dataset._t
        y, t = ensembler.predict_proba(test_X, test_X2), test_t
        y_pred = y > ensembler.threshold
        y_pred_cv = y > ensembler.threshold_cv
        result = classification_metrics(y_pred, t)
        result_cv = classification_metrics(y_pred_cv, t)
        result_theoretical = classification_metrics(y, t)
        scores.update(dict(
            test_fbeta=result['fbeta'],
            test_fbeta_cv=result_cv['fbeta'],
            test_fbeta_theoretical=result_theoretical['fbeta'],
            test_threshold_theoretical=result_theoretical['threshold'],
        ))

    print(scores)

    # Predict submit datasets
    submit_y = ensembler.predict(submit_dataset.X, submit_dataset.X2)
    submit_df['prediction'] = submit_y
    submit_df = submit_df[['qid', 'prediction']]
    submit_df.to_csv(config.outdir / 'submission.csv', index=False)

    return scores


if __name__ == '__main__':
    main()
