def by_auc(y, y1, auc1, y2, auc2):
    """
    m/n 代表正负样本，通常负样本更多
    :return 正样本数
    """
    import pandas as pd
    assert auc1 != auc2
    sum_positive_r1, sum_positive_r2 = pd.DataFrame(
        y, columns=['y']).assign(
            y1=pd.Series(y1).rank(), y2=pd.Series(y2).rank()).groupby(
                'y')['y1', 'y2'].sum()[lambda x: x.index == 1].values[0]

    m_plus_n = len(y)
    m_multi_n = (sum_positive_r1 - sum_positive_r2) / (auc1 - auc2)

    delta = (m_plus_n**2 - 4 * m_multi_n)**0.5
    _ = (m_plus_n + delta) / 2, (m_plus_n - delta) / 2
    m, n = min(_), max(_)
    print('正样本数：%f' % m)
    print('负样本数：%f' % n)
    print(f'正负样本比：1 / {n / m:f}')
