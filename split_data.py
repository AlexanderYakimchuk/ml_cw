import numpy as np
import pandas as pd
import os

dirname = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(dirname, 'data/heart.csv'))


def split(df):
    data = df.values
    bound = int(data.shape[0] * 0.8)

    np.random.shuffle(data)
    columns = list(df.columns)
    train_data = data[:bound, :]
    train_frame = pd.DataFrame(data=train_data,
                               columns=columns)
    train_frame = train_frame.astype(
        {name: 'int32' for name in set(columns) - {'oldpeak'}})

    test_data = data[bound:, :]
    test_frame = pd.DataFrame(data=test_data,
                              columns=columns)
    test_frame = test_frame.astype(
        {name: 'int32' for name in set(columns) - {'oldpeak'}})
    return train_frame, test_frame


if __name__ == "__main__":

    train_frame, test_frame = split(df)
    train_frame.to_csv(os.path.join(dirname, 'data/train.csv'), index=False)
    test_frame.to_csv(os.path.join(dirname, 'data/test.csv'), index=False)
