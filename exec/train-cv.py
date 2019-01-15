import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from gensim.corpora import Dictionary
from torch.utils.data import DataLoader
from tqdm import tqdm

import qiqc
from qiqc.builder import build_preprocessor
from qiqc.builder import build_tokenizer
from qiqc.builder import build_ensembler
from qiqc.builder import build_optimizer
from qiqc.datasets import load_qiqc
from qiqc.models import load_pretrained_vectors
from qiqc.model_selection import classification_metrics, ClassificationResult
from qiqc.utils import pad_sequence, set_seed, parallel_apply


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', '-m', type=Path, required=True)
    parser.add_argument('--device', '-g', type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--outdir', '-o', type=str, default='test')
    parser.add_argument('--optuna-trials', type=int)
    parser.add_argument('--gridsearch', action='store_true')
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--cv-part', type=int)
    parser.add_argument('--holdout', action='store_true')
    parser.add_argument('--processes', type=int, default=2)

    args = parser.parse_args(args)
    config = qiqc.config.build_config(args)
    outdir = Path('results') / '/'.join(args.modeldir.parts[1:])
    config['outdir'] = outdir / config['outdir']
    if args.test:
        config['n_rows'] = 500
        config['batchsize'] = 64
        config['epochs'] = 2
        config['cv_part'] = 2
    else:
        config['n_rows'] = None
    qiqc.utils.rmtree_after_confirmation(config['outdir'], args.test)

    if args.gridsearch:
        qiqc.model_selection.train_gridsearch(config, train)
    elif args.optuna_trials is not None:
        qiqc.model_selection.train_optuna(config, train)
    else:
        train(config)


def train(config):
    config['outdir'].mkdir(parents=True, exist_ok=True)
    modelconf = qiqc.loader.load_module(config['modeldir'] / 'model.py')
    build_models = modelconf.build_models
    build_sampler = modelconf.build_sampler

    print(config)
    start = time.time()
    set_seed(config['seed'])
    train_df, submit_df = load_qiqc(n_rows=config['n_rows'])
    preprocessor = build_preprocessor(config['preprocessors'])
    tokenizer = build_tokenizer(config['tokenizer'])

    print('Preprocess texts...')
    preprocess = lambda x: x.apply(lambda x: tokenizer(preprocessor(x)))  # NOQA
    train_df['tokens'] = parallel_apply(train_df.question_text, preprocess)
    submit_df['tokens'] = parallel_apply(submit_df.question_text, preprocess)
    all_df = pd.concat([train_df, submit_df], ignore_index=True, sort=False)

    print('Build vocabulary...')
    vocab = Dictionary(all_df.tokens.values, prune_at=None)
    dfs = sorted(vocab.dfs.items(), key=lambda x: x[1], reverse=True)
    token2id = dict(
        **{'<PAD>': 0},
        **dict([(vocab[idx], i + 1) for i, (idx, freq) in enumerate(dfs)]))
    word_freq = dict(
        **{'<PAD>': 1},
        **dict([(vocab[idx], freq) for idx, freq in dfs]))
    assert token2id['<PAD>'] == 0

    train_df['token_ids'] = train_df.tokens.apply(
        lambda xs: pad_sequence([token2id[x] for x in xs], config['maxlen']))
    submit_df['token_ids'] = submit_df.tokens.apply(
        lambda xs: pad_sequence([token2id[x] for x in xs], config['maxlen']))

    # Train : Test split for holdout training
    if config['holdout']:
        train_df, test_df = sklearn.model_selection.train_test_split(
            train_df, test_size=0.1, random_state=0)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        test_X = torch.Tensor(test_df.token_ids.tolist()).type(torch.long)
        test_X = test_X.to(config['device'])
        test_t = test_df.target[:, None]

    train_X = torch.Tensor(train_df.token_ids.tolist()).type(torch.long)
    train_W = torch.Tensor(train_df.weights).type(torch.float)
    train_t = torch.Tensor(train_df.target[:, None]).type(torch.float)

    submit_X = torch.Tensor(submit_df.token_ids).type(torch.long)
    submit_X = submit_X.to(config['device'])

    print('Load pretrained vectors and build models...')
    pretrained_vectors = load_pretrained_vectors(
        config['model']['embed']['src'], token2id, test=config['test'])
    models, unk_indices = build_models(
        config, word_freq, token2id, pretrained_vectors, all_df)

    print('Start training...')
    splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=config['cv'], shuffle=True, random_state=config['seed'])
    train_results, valid_results = [], []
    best_models = {}
    for i_cv, (train_indices, valid_indices) in enumerate(
            splitter.split(train_X, train_t)):
        if config['cv_part'] is not None and i_cv >= config['cv_part']:
            break
        _train_X = train_X[train_indices].to(config['device'])
        _train_W = train_W[train_indices].to(config['device'])
        _train_t = train_t[train_indices].to(config['device'])

        _valid_X = train_X[valid_indices].to(config['device'])
        _valid_W = train_W[valid_indices].to(config['device'])
        _valid_t = train_t[valid_indices].to(config['device'])

        train_dataset = torch.utils.data.TensorDataset(
            _train_X, _train_t, _train_W)
        valid_dataset = torch.utils.data.TensorDataset(
            _valid_X, _valid_t, _valid_W)
        valid_iter = DataLoader(
            valid_dataset, batch_size=config['batchsize_valid'])

        model = models.pop(0)
        model = model.to_device(config['device'])
        optimizer = build_optimizer(config['optimizer'], model)
        train_result = ClassificationResult(
            'train', config['outdir'], str(i_cv))
        valid_result = ClassificationResult(
            'valid', config['outdir'], str(i_cv))

        for epoch in range(config['epochs']):
            epoch_start = time.time()
            sampler = build_sampler(
                config['batchsize'], i_cv, epoch,
                train_df.weights[train_indices].values)
            train_iter = DataLoader(
                train_dataset, sampler=sampler, drop_last=True,
                batch_size=config['batchsize'], shuffle=sampler is None)

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

            # Validation loop
            for i, batch in enumerate(
                    tqdm(valid_iter, desc='valid', leave=False)):
                model.eval()
                loss, output = model.calc_loss(*batch)
                valid_result.add_record(**output)
            valid_result.calc_score(epoch)
            summary = pd.DataFrame([
                train_result.summary.iloc[-1],
                valid_result.summary.iloc[-1],
            ]).set_index('name')
            epoch_time = time.time() - epoch_start
            pbar = '#' * (i_cv + 1) + '-' * (config['cv'] - 1 - i_cv)
            tqdm.write(f'\n{pbar} cv: {i_cv} / {config["cv"]}, epoch {epoch}, '
                       f'time: {epoch_time}')
            tqdm.write(str(summary))

            # Case: updating the best score
            if epoch == valid_result.best_epoch:
                best_models[i_cv] = deepcopy(model)

        train_results.append(train_result)
        valid_results.append(valid_result)

    # Build ensembler
    ensembler = build_ensembler(config['ensembler']['model'])(
        config=config,
        models=list(best_models.values()),
        results=valid_results,
    )

    ensembler.fit(train_X, train_t, config['ensembler']['test_size'])
    scores = dict(
        valid_fbeta=np.array([r.best_fbeta for r in valid_results]).mean(),
        valid_epoch=np.array([r.best_epoch for r in valid_results]).mean(),
        threshold_cv=ensembler.threshold_cv,
        threshold=ensembler.threshold,
        elapsed_time=time.time() - start,
    )

    if config['holdout']:
        df = test_df
        y, t = ensembler.predict_proba(test_X), test_t
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

    if config['logging']:
        part = config['cv'] if config['cv_part'] is None else config['cv_part']
        indices = np.concatenate(
            [s[1] for s in splitter.split(train_X, train_t)][:part])
        df, t = train_df.iloc[indices].copy(), train_t.numpy()[indices]
        y = np.concatenate([r.best_ys for r in valid_results])
        y_pred = y > ensembler.threshold
        unk_freq = dict(np.array(list(word_freq.items()))[unk_indices])
        df['y'] = y - ensembler.threshold
        df['t'] = df.target
        maxlen = config['maxlen']
        df['_tokens'] = df.tokens.apply(lambda xs: [x in unk_freq for x in xs])
        df['_tokens'] = df.apply(
            lambda x: np.where(
                x._tokens[:maxlen], '<UNK>', x.tokens[:maxlen]), axis=1)
        is_error = y_pred != t
        tp = np.argwhere(~is_error * t.astype('bool'))[:, 0]
        fp = np.argwhere(is_error * ~t.astype('bool'))[:, 0]
        fn = np.argwhere(is_error * t.astype('bool'))[:, 0]
        df = df[['qid', 'question_text', 'tokens', '_tokens', 'y', 't']]
        df.iloc[tp].to_csv(config['outdir'] / 'TP.tsv', sep='\t')
        df.iloc[fp].to_csv(config['outdir'] / 'FP.tsv', sep='\t')
        df.iloc[fn].to_csv(config['outdir'] / 'FN.tsv', sep='\t')
        json.dump(
            word_freq, open(config['outdir'] / 'word.json', 'w'), indent=4)
        json.dump(
            unk_freq, open(config['outdir'] / 'unk.json', 'w'), indent=4)
        for i, result in enumerate(valid_results):
            result.summary.to_csv(
                config['outdir'] / 'summary_valid_{i}.tsv', sep='\t')

    print(scores)

    # Predict submit datasets
    submit_y = ensembler.predict(submit_X)
    submit_df['prediction'] = submit_y
    submit_df = submit_df[['qid', 'prediction']]
    submit_df.to_csv(config['outdir'] / 'submission.csv', index=False)

    return scores


if __name__ == '__main__':
    main()
