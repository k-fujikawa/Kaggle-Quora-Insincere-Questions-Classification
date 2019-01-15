import pandas as pd
from sklearn.model_selection import ParameterGrid

import qiqc


def train_gridsearch(config, train):
    hypconf = qiqc.loader.load_module(config['modeldir'] / 'gridsearch.py')
    hyperparams = hypconf.build_gridsearch_params()
    gridparams = pd.DataFrame(list(ParameterGrid(hyperparams)))
    outdir = config['outdir']

    for i, _hyperparams in gridparams.iterrows():
        for keys, param in _hyperparams.items():
            qiqc.config.set_by_path(config, keys.split('.'), param)
        name = ' '.join([f'{k}:{p}' for k, p in _hyperparams.items()])
        config['outdir'] = outdir / name
        print(f'\nExperiment {i+1}/{len(gridparams)}: {name}')
        results = train(config)
        for k, v in results.items():
            gridparams.ix[i, k] = v

    if config['holdout']:
        keys = ['test_fbeta', 'valid_fbeta']
    else:
        keys = ['valid_fbeta']

    scores = gridparams.sort_values(keys, ascending=False)
    scores.to_csv(f'{outdir}/result.tsv', sep='\t')
    if 'seed' in hyperparams:
        names = [k for k in hyperparams.keys() if k != 'seed']
        if len(names) > 0:
            scores = scores.groupby(names).agg(['mean', 'std'])
            print(scores)
        else:
            print(scores.describe())
