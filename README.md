<h1 align = "center">:rocket: TQL :facepunch:</h1>

---
> 太强了：数据不是创造规律,而是展示那些原本的规律

---
## Install
```
pip install tql
```
## Usage
#### `from tql.pipe import *`
```python
@X
def xfunc1(x):
    _ = x.split()
    print(_)
    return _
@X
def xfunc2(x):
    _ = '>>'.join(x)
    print(_)
    return _

'I am very like a linux pipe' | xfunc1 | xfunc2
```
- `xtqdm`

    ![tqdm](Pictures/tqdm.png)

- `xfilter / xmap / xsort / xreduce`
```python
iterable | xfilter(lambda x: len(x) > 1) | xmap(str.upper) | xsort | xreduce(lambda x, y: x + '-' + y)

'AM-LIKE-LINUX-PIPE-VERY'
```

- `xsummary`
```python
iterable | xsummary

counts               7
uniques              7
missing              0
missing_perc        0%
types           unique
Name: iterable, dtype: object
```
- ...

---
