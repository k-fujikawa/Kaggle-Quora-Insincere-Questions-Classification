import pandas as pd


def load_qiqc(n_rows=None):
    train_df = pd.read_csv('/src/input/train.csv', nrows=n_rows)
    submit_df = pd.read_csv('/src/input/test.csv', nrows=n_rows)
    n_labels = {
        0: (train_df.target == 0).sum(),
        1: (train_df.target == 1).sum(),
    }
    train_df['target'] = train_df.target.astype('f')
    train_df['weights'] = train_df.target.apply(lambda t: 1 / n_labels[t])

    return train_df, submit_df
