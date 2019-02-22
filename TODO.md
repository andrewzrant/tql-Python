`⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹`

---
[ELMO][1]
---
```
top10_sellers = data.pivot_table(values='Purchase',index=['Product_ID'], aggfunc='count').reset_index().sort_values(by = 'Purchase',ascending=False).head(10)

from mlxtend.frequent_patterns import apriori, association_rules
df = pd.DataFrame([[1, 1], [1, 0]], columns=['a', 'b'])
association_rules(apriori(df))

https://www.kesci.com/home/project/5be7e948954d6e0010632ef2
```

[1]: https://blog.csdn.net/sinat_26917383/article/details/81913790

https://github.com/Jie-Yuan/DataMining/tree/master/0_DA/udfs
