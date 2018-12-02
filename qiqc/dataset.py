from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler


def split_holdout(dataset, batch_size, batch_size_valid, n_splits=1,
                  test_size=0.1, random_state=0, balancing=False):
    splitter = ShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state)
    train_iters, test_iters = [], []

    for train_indices, test_indices in splitter.split(dataset):
        if balancing:
            sampler = WeightedRandomSampler(
                weights=dataset.df.weights[train_indices],
                num_samples=len(train_indices),
                replacement=True,
            )
        else:
            sampler = None

        train_iters.append(DataLoader(
            Subset(dataset, train_indices),
            sampler=sampler, drop_last=True,
            batch_size=batch_size))
        test_iters.append(DataLoader(
            Subset(dataset, test_indices),
            batch_size=batch_size_valid, shuffle=False))

    return train_iters, test_iters
