import argparse
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
from qiqc.datasets import load_qiqc
from qiqc.embeddings import build_word_vectors
from qiqc.embeddings import load_pretrained_vectors
from qiqc.model_selection import classification_metrics, ClassificationResult
from qiqc.utils import pad_sequence, set_seed


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', '-m', type=Path, required=True)
    parser.add_argument('--device', '-g', type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--outdir', '-o', type=str, default='test')
    parser.add_argument('--batchsize', '-b', type=int, default=512)
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
    modelconf = qiqc.loader.load_module(config['modeldir'] / 'model.py')
    build_embedding = modelconf.build_embedding
    build_model = modelconf.build_model
    build_sampler = modelconf.build_sampler
    build_optimizer = modelconf.build_optimizer

    print(config)
    set_seed(config['seed'])
    train_df, submit_df = load_qiqc(n_rows=config['n_rows'])
    preprocessor = build_preprocessor(config['preprocessors'])
    tokenizer = build_tokenizer(config['tokenizer'])

    print('Preprocess texts...')
    train_df['tokens'] = train_df.question_text.apply(
        lambda x: tokenizer(preprocessor(x)))
    submit_df['tokens'] = submit_df.question_text.apply(
        lambda x: tokenizer(preprocessor(x)))
    tokens = train_df.tokens.append(submit_df.tokens).values

    print('Build vocabulary...')
    vocab = Dictionary(tokens, prune_at=None)
    dfs = sorted(vocab.dfs.items(), key=lambda x: x[1], reverse=True)
    token2id = dict(
        **{'<PAD>': 0},
        **dict([(f'<{i}>', i) for i in range(1, config['maxlen'] + 1)]),
        **dict([(vocab[idx], i + config['maxlen'] + 1)
                for i, (idx, freq) in enumerate(dfs)]))
    word_freq = dict([(vocab[idx], freq) for idx, freq in dfs])
    assert token2id['<PAD>'] == 0

    train_df['token_ids'] = train_df.tokens.apply(
        lambda xs: pad_sequence([token2id[x] for x in xs], config['maxlen']))
    submit_df['token_ids'] = submit_df.tokens.apply(
        lambda xs: pad_sequence([token2id[x] for x in xs], config['maxlen']))
    if config['embedding']['add_los']:
        def replace_to_los(x):
            length = (x != 0).sum()
            x[length - 1] = length
            return x
        train_df['token_id'] = train_df.token_ids.apply(replace_to_los)
        submit_df['token_id'] = submit_df.token_ids.apply(replace_to_los)

    print('Load pretrained vectors...')
    pretrained_vectors = load_pretrained_vectors(
        config['embedding']['src'], token2id, test=config['test'])
    qiqc_vectors = []
    for name, _pretrained_vectors in pretrained_vectors.items():
        vec, unk_freq = build_word_vectors(
            word_freq, _pretrained_vectors, config['vocab']['min_count'])
        print(f'UNK of {name} <high>: {len(unk_freq)} / {len(word_freq)}'
              f' ({100 * len(unk_freq) / len(word_freq):.2f} %)')
        print(f'    ex. {list(unk_freq.items())[:10]}')
        qiqc_vectors.append(vec)
    qiqc_vectors = np.array(qiqc_vectors).mean(axis=0)

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

    print('Start training...')
    splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=config['cv'], shuffle=True, random_state=config['seed'])
    train_results, valid_results = [], []
    best_models = {}
    start = time.time()
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

        embedding = build_embedding(
            i_cv, config, tokens, word_freq, token2id, qiqc_vectors)
        model = build_model(i_cv, config, embedding)
        model = model.to_device(config['device'])
        optimizer = build_optimizer(i_cv, config, model)
        train_result = ClassificationResult('train', config['outdir'])
        valid_result = ClassificationResult('valid', config['outdir'])

        for epoch in range(config['epochs']):
            epoch_start = time.time()
            sampler = build_sampler(
                i_cv, epoch, train_df.weights[train_indices].values)
            train_iter = DataLoader(
                train_dataset, sampler=sampler, drop_last=True,
                batch_size=config['batchsize'], shuffle=sampler is None)

            # Training loop
            for batch in tqdm(train_iter, desc='train', leave=False):
                model.train()
                optimizer.zero_grad()
                loss, output = model.calc_loss(*batch)
                loss.backward()
                optimizer.step()
                train_result.add_record(**output)
            train_result.calc_score(epoch)

            # Validation loop
            for batch in tqdm(valid_iter, desc='valid', leave=False):
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
    ensembler = build_ensembler(config['ensembler'])(
        models=list(best_models.values()),
        results=valid_results,
        device=config['device'],
        batchsize_train=config['batchsize'],
        batchsize_valid=config['batchsize_valid'],
    )

    indices = np.random.permutation(range(int(len(train_X) * 0.1)))
    X, t = train_X[indices], train_t[indices].numpy()
    y, ensemble_score = ensembler.fit(X, t)
    y_pred = y > ensemble_score['threshold']
    df = train_df.iloc[indices].reset_index(drop=True)

    scores = dict(
        valid_fbeta=np.array([r.best_fbeta for r in valid_results]).mean(),
        valid_epoch=np.array([r.best_epoch for r in valid_results]).mean(),
        valid_threshold=np.array([
            r.best_threshold for r in valid_results]).mean(),
        ensemble_fbeta=ensemble_score['fbeta'],
        ensemble_threshold=ensemble_score['threshold'],
        elapsed_time=time.time() - start,
    )

    if config['holdout']:
        df = test_df
        y, t = ensembler.predict_proba(test_X), test_t
        y_pred = y > ensemble_score['threshold']
        result = classification_metrics(y_pred, t)
        result_theoretical = classification_metrics(y, t)
        scores.update(dict(
            test_fbeta=result['fbeta'],
            test_fbeta_theoretical=result_theoretical['fbeta'],
            test_threshold=result['threshold'],
            test_threshold_theoretical=result_theoretical['threshold'],
        ))

    if config['logging']:
        df['y'] = y - ensembler.threshold
        df['t'] = df.target
        maxlen = config['maxlen']
        df['_tokens'] = df.apply(
            lambda x: embedding[x.token_ids].all(axis=1)[:len(x.tokens)],
            axis=1)
        df['_tokens'] = df.apply(
            lambda x: np.where(x._tokens, x.tokens[:maxlen], '<UNK>'), axis=1)
        is_error = y_pred != t
        tp = np.argwhere(~is_error * t.astype('bool'))[:, 0]
        tn = np.argwhere(~is_error * ~t.astype('bool'))[:, 0]
        fp = np.argwhere(is_error * ~t.astype('bool'))[:, 0]
        fn = np.argwhere(is_error * t.astype('bool'))[:, 0]
        df = df[['qid', 'question_text', 'tokens', '_tokens', 'y', 't']]
        df.iloc[tp].to_csv(f'{config["outdir"]}/TP.tsv', sep='\t')
        df.iloc[tn].to_csv(f'{config["outdir"]}/TN.tsv', sep='\t')
        df.iloc[fp].to_csv(f'{config["outdir"]}/FP.tsv', sep='\t')
        df.iloc[fn].to_csv(f'{config["outdir"]}/FN.tsv', sep='\t')

    print(scores)

    # Predict submit datasets
    submit_y = ensembler.predict(submit_X)
    submit_df['prediction'] = submit_y
    submit_df = submit_df[['qid', 'prediction']]

    return scores


if __name__ == '__main__':
    main()
