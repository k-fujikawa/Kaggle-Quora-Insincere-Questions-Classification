import operator
from functools import reduce
from pathlib import Path

import yaml


def build_config(args):
    confpath = Path(args.modeldir) / 'config.yml'
    config = yaml.load(open(confpath))
    config = {**config, **vars(args)}
    return config


def get_by_path(config, items):
    return reduce(operator.getitem, items, config)


def set_by_path(config, items, value):
    get_by_path(config, items[:-1])[items[-1]] = value
