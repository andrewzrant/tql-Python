"""
auc表示（正样本概率>负样本概率）的概率 # 可以算期望

根据线上auc猜测正负样本数
1/正样本数 = 2 * auc - 1  # gini = 2 * auc - 1 与gini公式一致
如果auc<0.5就是负样本数

预测错的个数 > (1 - auc) * 正样本数 # 至少 (1 - auc) * 正样本数 个
"""
