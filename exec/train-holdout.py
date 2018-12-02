import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import torch
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader

import qiqc
from qiqc.datasets import QIQCTrainDataset, QIQCSubmitDataset


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', '-m', type=str, required=True)
    parser.add_argument('--device', '-g', type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--outdir', '-o', type=str, default='results')
    parser.add_argument('--batchsize', '-b', type=int, default=512)
    parser.add_argument('--optuna-trials', type=int)
    parser.add_argument('--gridsearch', action='store_true')

    args = parser.parse_args(args)
    config = qiqc.config.build_config(args)
    qiqc.set_seed(config['model_seed'])
    if args.test:
        config['outdir'] = str(Path(args.modeldir) / 'test')
        config['n_rows'] = 1000
    else:
        config['outdir'] = Path(args.modeldir) / config['outdir']
        config['n_rows'] = None
    qiqc.utils.rmtree_after_confirmation(config['outdir'], args.test)

    if args.gridsearch:
        gridsearch(config)
    elif args.optuna_trials is not None:
        train_optuna(config)
    else:
        train(config)


def gridsearch(config):
    hypconf = qiqc.loader.load_module(
        Path(config['modeldir']) / 'gridsearch.py')
    hyperparams = hypconf.build_gridsearch_params()
    gridparams = pd.DataFrame(list(ParameterGrid(hyperparams)))
    outdir = config['outdir']

    for i, _hyperparams in gridparams.iterrows():
        for keys, param in _hyperparams.items():
            qiqc.config.set_by_path(config, keys.split('.'), param)
        name = ' '.join([f'{k}:{p}' for k, p in _hyperparams.items()])
        config['outdir'] = f'{outdir}/{name}'
        print(f'\nExperiment {i+1}/{len(gridparams)}: {name}')
        results = pd.DataFrame(train(config))
        gridparams.ix[i, 'score'] = results['bestscore'].mean()
        gridparams.ix[i, 'epoch'] = results['bestepoch'].mean()
        gridparams.ix[i, 'time'] = results['time'].mean()

    scores = gridparams.sort_values('score', ascending=False)
    scores.to_csv(f'{outdir}/result.tsv', sep='\t')
    print(scores)


def train_optuna(config):
    study = optuna.create_study()
    study.optimize(
        partial(optuna_objective, config=config),
        n_trials=config['optuna_trials'])

    trial = study.best_trial
    pruned_trials = [t for t in study.trials
                     if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials
                       if t.state == optuna.structs.TrialState.COMPLETE]

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    outdir = config['outdir']
    study.trials_dataframe().to_csv(f'{outdir}/result.tsv', sep='\t')


def optuna_objective(trial, config):
    hypconf = qiqc.loader.load_module(
        Path(config['modeldir']) / 'optuna.py')
    _config = hypconf.build_optuna_config(trial, config)
    results = train(_config, trial=trial)
    score = np.array([r['bestscore'] for r in results]).mean()
    return -1 * score


def train(config, trial=None):
    print(config)
    train_rawdata = QIQCTrainDataset(nrows=config['n_rows'])
    submit_rawdata = QIQCSubmitDataset(nrows=config['n_rows'])
    modelconf = qiqc.loader.load_module(
        Path(config['modeldir']) / 'model.py')
    preprocessor = modelconf.build_preprocessor(config)

    train_rawdata.df['tokens'] = train_rawdata.texts.apply(preprocessor)
    submit_rawdata.df['tokens'] = submit_rawdata.texts.apply(preprocessor)
    tokens = np.concatenate(
        [train_rawdata.df.tokens.values, submit_rawdata.df.tokens.values])
    featurizer = modelconf.build_featurizer(config, tokens)
    config['model']['embedding_matrix'] = featurizer.vectors
    train_rawdata.df['target'] = train_rawdata.df.target.apply(
        lambda x: torch.tensor(x, dtype=torch.float))
    train_rawdata.df['token_ids'] = train_rawdata.df.tokens.apply(featurizer)
    submit_rawdata.df['token_ids'] = submit_rawdata.df.tokens.apply(featurizer)

    train_rawdata, test_rawdata = train_rawdata.train_test_split(
        test_size=0.1, random_state=0)
    train_iters, valid_iters = qiqc.dataset.split_holdout(
        train_rawdata, config['batchsize'], config['batchsize_valid'],
        random_state=config['dataset_seed'], n_splits=1,
        balancing=config['balancing'])
    test_iter = DataLoader(
        test_rawdata, batch_size=config['batchsize_valid'],
        shuffle=False)

    results = []
    for train_iter, valid_iter in zip(train_iters, valid_iters):
        model = modelconf.build_model(config).to_device(config['device'])
        optimizer = modelconf.build_optimizer(config, model)
        trainer = qiqc.trainer.Trainer(
            model, optimizer, featurizer, config['device'], config['outdir'])
        result = trainer.train(
            config, train_iter, valid_iter, test_iter, trial=trial)
        results.append(result)
        print(result['bestscore'])
    del config['model']['embedding_matrix']

    return results


if __name__ == '__main__':
    main()
