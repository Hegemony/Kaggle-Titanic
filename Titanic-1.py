import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('F:/anaconda3/Lib/site-packages')  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

train_data = pd.read_csv('./Kaggle Dataset/Titanic/train.csv')  # (891, 12)
test_data = pd.read_csv('./Kaggle Dataset/Titanic/test.csv')   # (418, 11)

print(train_data.shape, test_data.shape)

all_features = pd.concat((train_data.iloc[:, 2:-1], test_data.iloc[:, 1:]))
'''
iloc主要使用数字来索引数据，而不能使用字符型的标签来索引数据。而loc则刚好相反，只能使用字符型标签来索引数据，
不能使用数字来索引数据，不过有特殊情况，当数据框dataframe的行标签或者列标签为数字，loc就可以来其来索引。
'''
print(all_features.shape)  # (1309, 11)

print(all_features.dtypes)
print('-'*100)

"""
预处理数据
我们对连续数值的特征做标准化（standardization）：设该特征在整个数据集上的均值为μ，标准差为σ。那么，
我们可以将该特征的每个值先减去μ再除以σ得到标准化后的每个特征值。对于缺失的特征值，我们将其替换成该特征的均值。
"""
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# Index对象保存着索引标签数据，它可以快速找到标签对应的整数下标，其功能与Python的字典类似
print(numeric_features)

all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
'''
pandas 的 apply() 函数可以作用于 Series 或者整个 DataFrame，功能也是自动遍历整个 Series 或者 DataFrame, 对每一个元素运行指定的函数。
'''
# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)
'''
fillna()会填充nan数据，返回填充后的结果。如果希望在原DataFrame中修改，则把inplace设置为True
'''

# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)   # (1309, 2438)


"""
最后，通过values属性得到NumPy格式的数据，并转成Tensor方便后面的训练。
"""
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.Survived.values, dtype=torch.float).view(-1)


loss = torch.nn.CrossEntropyLoss()


def get_net(feature_num):
    net = torch.nn.Sequential(
        torch.nn.Linear(feature_num, 2),
        # # 完成从输入层到隐藏层的线性变换
        # torch.nn.ReLU(),
        # 经过激活函数
        # torch.nn.Linear(feature_num // 2, 2),
        # torch.nn.ReLU()
    )
    for param in net.parameters():
        # print(param)
        nn.init.normal_(param, mean=0, std=0.01)
    return net


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.long())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(loss(net(train_features), train_labels.long()).item())
        if test_labels is not None:
            test_ls.append(loss(net(test_features), test_labels.long()).item())
    return train_ls, test_ls


"""
K折交叉验证:
我们在3.11节（模型选择、欠拟合和过拟合）中介绍了K折交叉验证。它将被用来选择模型设计并调节超参数。
下面实现了一个函数，它返回第i折交叉验证时所需要的训练和验证数据
"""
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


'''
画图
'''
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    # set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


"""
在K折交叉验证中我们训练K次并返回训练和验证的平均误差。
"""
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'CrossEntropy',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train CrossEntropy %f, valid CrossEntropy %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.01, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train CrossEntropy %f, avg valid CrossEntropy %f' % (k, train_l, valid_l))


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'CrossEntropy')
    print('train CrossEntropy %f' % train_ls[-1])
    preds = net(test_features).argmax(dim=1).detach().numpy()
    # print(preds, preds.shape)
    # test_data['Survived'] = pd.Series(preds.reshape(1, -1)[0])
    test_data['Survived'] = pd.Series(preds)
    submission = pd.concat([test_data['PassengerId'], test_data['Survived']], axis=1)
    submission.to_csv('./submission.csv', index=False)

# train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)