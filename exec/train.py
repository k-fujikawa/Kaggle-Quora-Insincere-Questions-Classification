import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from torch.utils.data import DataLoader
from tqdm import tqdm

import qiqc
from qiqc.builder import build_preprocessor
from qiqc.builder import build_tokenizer
from qiqc.builder import build_ensembler
from qiqc.builder import build_optimizer
from qiqc.features import load_pretrained_vectors, WordVocab
from qiqc.datasets import load_qiqc, QIQCDataset
from qiqc.model_selection import classification_metrics, ClassificationResult
from qiqc.preprocessors import PreprocessPipeline
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
        config['epochs'] = 3
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
    sent2vec = modelconf.build_sentence_feature()

    print(config)
    start = time.time()
    set_seed(config['seed'])
    train_df, submit_df = load_qiqc(n_rows=config['n_rows'])
    preprocessor = build_preprocessor(config['preprocessors'])
    tokenizer = build_tokenizer(config['tokenizer'])
    train_dataset = QIQCDataset(train_df)
    submit_dataset = QIQCDataset(submit_df)

    print('Preprocess texts...')
    tokenize = PreprocessPipeline(preprocessor, tokenizer)
    train_dataset.preprocess(
        'question_text', 'tokens', tokenize, parallel_apply)
    submit_dataset.preprocess(
        'question_text', 'tokens', tokenize, parallel_apply)

    if sent2vec is not None:
        print('Extract sentence features...')
        applyfunc = lambda x, func: x.apply(func)  # NOQA
        train_dataset.preprocess(
            'question_text', '_X2', sent2vec.extract_features, applyfunc)
        submit_dataset.preprocess(
            'question_text', '_X2', sent2vec.extract_features, applyfunc)
        train_dataset.preprocess(
            '_X2', 'X2', sent2vec.fit_transform, applyfunc)
        submit_dataset.preprocess(
            '_X2', 'X2', sent2vec.transform, applyfunc)

    print('Build vocabulary...')
    vocab = WordVocab()
    vocab.add_documents(train_dataset.positives.tokens, 'pos')
    vocab.add_documents(train_dataset.negatives.tokens, 'neg')
    vocab.add_documents(submit_dataset.df.tokens, 'submit')
    vocab.build()

    print('Build token ids...')
    applyfunc = lambda x, func: x.apply(func)  # NOQA
    token2id = lambda xs: pad_sequence(  # NOQA
        [vocab.token2id[x] for x in xs], config['maxlen'])
    train_dataset.preprocess(
        'tokens', 'token_ids', token2id, applyfunc)
    submit_dataset.preprocess(
        'tokens', 'token_ids', token2id, applyfunc)

    # Train : Test split for holdout training
    if config['holdout']:
        train_df, test_df = sklearn.model_selection.train_test_split(
            train_dataset.df, test_size=0.1, random_state=0)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        train_dataset = QIQCDataset(train_df)
        test_dataset = QIQCDataset(test_df)
        test_dataset.build(config['device'])

    train_dataset.build(config['device'])
    submit_dataset.build(config['device'])

    print('Load pretrained vectors and build models...')
    all_df = pd.concat(
        [train_dataset.df, submit_dataset.df], ignore_index=True, sort=False)
    pretrained_vectors = load_pretrained_vectors(
        config['model']['embed']['src'], vocab.token2id, test=config['test'])
    models, unk_indices = build_models(
        config, vocab, pretrained_vectors, all_df)

    print('Start training...')
    splitter = sklearn.model_selection.StratifiedKFold(
        n_splits=config['cv'], shuffle=True, random_state=config['seed'])
    train_results, valid_results = [], []
    best_models = []
    for i_cv, (train_indices, valid_indices) in enumerate(
            splitter.split(train_dataset.df, train_dataset.df.target)):
        if config['cv_part'] is not None and i_cv >= config['cv_part']:
            break
        train_tensor = train_dataset.build_labeled_dataset(train_indices)
        valid_tensor = train_dataset.build_labeled_dataset(valid_indices)
        valid_iter = DataLoader(
            valid_tensor, batch_size=config['batchsize_valid'])

        model = models.pop(0)
        model = model.to_device(config['device'])
        model_snapshots = []
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
                train_tensor, sampler=sampler, drop_last=True,
                batch_size=config['batchsize'], shuffle=sampler is None)
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
            if epoch >= config['validate_from']:
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
            pbar = '#' * (i_cv + 1) + '-' * (config['cv'] - 1 - i_cv)
            tqdm.write(f'\n{pbar} cv: {i_cv} / {config["cv"]}, epoch {epoch}, '
                       f'time: {epoch_time}')
            tqdm.write(str(summary))

        train_results.append(train_result)
        valid_results.append(valid_result)
        best_indices = valid_result.summary.fbeta.argsort()[::-1]
        best_models.extend([model_snapshots[i] for i in
                            best_indices[:config['ensembler']['n_snapshots']]])

    # Build ensembler
    ensembler = build_ensembler(config['ensembler']['model'])(
        config=config,
        models=best_models,
        results=valid_results,
    )

    train_X, train_X2, train_t = \
        train_dataset.X, train_dataset.X2, train_dataset.t
    ensembler.fit(train_X, train_X2, train_t, config['ensembler']['test_size'])
    scores = dict(
        valid_fbeta=np.array([r.best_fbeta for r in valid_results]).mean(),
        valid_epoch=np.array([r.best_epoch for r in valid_results]).mean(),
        threshold_cv=ensembler.threshold_cv,
        threshold=ensembler.threshold,
        elapsed_time=time.time() - start,
    )

    if config['holdout']:
        test_X, test_X2, test_t = \
            test_dataset.X, test_dataset.X2, test_dataset.t
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

    if config['logging']:
        part = config['cv'] if config['cv_part'] is None else config['cv_part']
        indices = np.concatenate(
            [s[1] for s in splitter.split(train_X, train_t)][:part])
        df, t = train_df.iloc[indices].copy(), train_t.numpy()[indices]
        y = np.concatenate([r.best_ys for r in valid_results])
        y_pred = y > ensembler.threshold
        word_freq = vocab.word_freq
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
                config['outdir'] / f'summary_valid_{i}.tsv', sep='\t')

    print(scores)

    # Predict submit datasets
    submit_y = ensembler.predict(submit_dataset.X, submit_dataset.X2)
    submit_df['prediction'] = submit_y
    submit_df = submit_df[['qid', 'prediction']]
    submit_df.to_csv(config['outdir'] / 'submission.csv', index=False)

    return scores


if __name__ == '__main__':
    main()
