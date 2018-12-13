# X-Plan
```
import functools
import itertools
import socket
import sys
from contextlib import closing
from collections import deque

class Pipe:
    def __init__(self, function):
        self.function = function
        functools.update_wrapper(self, function)

    def __ror__(self, other):
        return self.function(other)

    def __call__(self, *args, **kwargs):
        return Pipe(lambda x: self.function(x, *args, **kwargs))
```
