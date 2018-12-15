def build_gridsearch_params():
    params = {}
    # params['featurizer.pretrain'] = ['gnews']
    # params['featurizer.n_finetune'] = [1, 2]
    # params['model.embed.alpha'] = [1e-5, 1e-4]
    params['model.encoder.n_layers'] = [2, 3]
    params['model.encoder.n_hidden'] = [64, 128]
    # params['model.mlp.bn'] = [True, False]
    params['model.embed.position'] = [True, False]
    params['model.embed.n_hidden'] = [0, 128]

    return params


def build_optuna_config(trial, config):
    # Hyperparams for Word embedding
    config['featurizer']['n_finetune'] = trial.suggest_int(
        'embed:finetune', 0, 3)

    # Hyperparams for Encoder
    config['model']['embed']['alpha'] = trial.suggest_loguniform(
        'embed:alpha', 1e-5, 1e-2)

    # Hyperparams for optimizer
    config['optimizer']['lr'] = trial.suggest_loguniform(
        'optimizer:lr', 1e-5, 1e-2)

    return config
