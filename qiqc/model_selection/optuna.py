from functools import partial

import numpy as np
import optuna

import qiqc


def train_optuna(config, train):
    study = optuna.create_study()
    study.optimize(
        partial(optuna_objective, config=config, train=train),
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


def optuna_objective(trial, config, train):
    hypconf = qiqc.loader.load_module(config['modeldir'] / 'optuna.py')
    _config = hypconf.build_optuna_config(trial, config)
    results = train(_config, trial=trial)
    score = np.array([r['bestscore'] for r in results]).mean()
    return -1 * score
