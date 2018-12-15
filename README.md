# X-Plan
```
import functools
import itertools
import socket
import sys
from contextlib import closing
from collections import deque

class X:
    def __init__(self, function):
        self.function = function
        functools.update_wrapper(self, function)

    def __ror__(self, other):
        return self.function(other)

    def __call__(self, *args, **kwargs):
        return Pipe(lambda x: self.function(x, *args, **kwargs))
        
@X
def xjson(dict_, ):
    print(json.dumps(dict_, indent=4))
#     return json.dumps(dict_, indent=4)

```
```
base_dir = os.path.dirname(os.path.realpath('__file__'))

```
sklearn.ex...

https://stackoverflow.com/questions/40287657/load-pickled-object-in-different-file-attribute-error
