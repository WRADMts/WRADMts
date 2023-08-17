import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler


# max min(0-1)
def norm(train, test):

    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train) # scale training data to [0,1] range
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)

    return train_ret, test_ret


# downsample by 10
def downsample_test(data, labels, down_len):
    np_data = np.array(data)
    np_labels = np.array(labels)

    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)


    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    # d_labels = np.max(d_labels, axis=1)
    d_labels = np.median(d_labels, axis=1)//1
    # d_labels = np.round(np.max(d_labels, axis=1))


    d_data = d_data.transpose()

    return d_data.tolist(), d_labels.tolist()

def downsample_train(data, down_len):
    np_data = np.array(data)

    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)


    d_data = d_data.transpose()

    return d_data.tolist()


def main():

    test = pd.read_csv('./data/ecg/test.csv', index_col=0)
    train = pd.read_csv('./data/ecg/train.csv', index_col=0)


    test = test.iloc[:, 0:]
    train = train.iloc[:, 0:]

    train = train.fillna(train.mean())
    test = test.fillna(test.mean())
    train = train.fillna(0)
    test = test.fillna(0)

    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())

    print(len(test.columns),test.columns)
    print(len(train.columns),train.columns)


    test_labels = test.attack

    test_labels = test.values[:,-1]

    test = test.drop(columns=['attack'])

    cols = [x for x in train.columns] # remove column name prefixes
    train.columns = cols
    test.columns = cols

    x_train, x_test = norm(train.values, test.values)
    # x_test = norm(test.values)

    for i, col in enumerate(test.columns):
        train.loc[:, col] = x_train[:, i]
        test.loc[:, col] = x_test[:, i]


    d_train_x = downsample_train(train.values, 10)
    d_test_x, d_test_labels = downsample_test(test.values, test_labels, 10)

    train_df = pd.DataFrame(d_train_x, columns = train.columns)
    test_df = pd.DataFrame(d_test_x, columns = test.columns)

    test_df['attack'] = d_test_labels

    print(train_df.values.shape)
    print(test_df.values.shape)


    train_df.to_csv('./data/ecg/train.csv')
    test_df.to_csv('./data/ecg/test.csv')

    f = open('./data/ecg/list.txt', 'w')
    for col in train.columns:
        f.write(col+'\n')
    f.close()

if __name__ == '__main__':
    main()
