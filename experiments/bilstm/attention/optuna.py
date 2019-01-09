def build_optuna_config(trial, config):
    config['model']['embed']['dropout'] = trial.suggest_uniform(
        'model.embed.dropout', 0.0, 1.0)
    config['model']['encoder']['n_hidden'] = int(trial.suggest_loguniform(
        'model.encoder.n_hidden', 32, 256))
    config['model']['mlp']['n_hidden'] = int(trial.suggest_loguniform(
        'model.mlp.n_hidden', 32, 256))

    return config
