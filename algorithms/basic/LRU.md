### 1. Use python OrderedDict to implement LRUCache

```python
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity=128):
        self.od = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.od:
            val = self.od[key]
            self.od.move_to_end(key)
            return val
        else:
            return None

    def put(self, key, val):
        if key in self.od:
            del self.od[key]
            self.od[key] = val
            # the other implementation
            # self.od[key] = val
            # self.od.move_to_end(key)
        else:
            self.od[key] = val
            if len(self.od) > self.capacity:
                self.od.popitem(last=False)

```
