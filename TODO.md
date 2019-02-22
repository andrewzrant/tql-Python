`⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹`

---
[ELMO][1]
---
```
top10_sellers = data.pivot_table(values='Purchase',index=['Product_ID'], aggfunc='count').reset_index().sort_values(by = 'Purchase',ascending=False).head(10)

```

[1]: https://blog.csdn.net/sinat_26917383/article/details/81913790

https://github.com/Jie-Yuan/DataMining/tree/master/0_DA/udfs
