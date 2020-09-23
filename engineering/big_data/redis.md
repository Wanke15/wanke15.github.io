### 最佳实践

1. 鉴于Redis的特性，应该**为所有的key设置TTL**

2. eviction policy 由原来的 volatile-lru 修改为 **allkeys-lru**。因为 volatile-lru 只作用于设置了TTL的key，而 allkeys-lru 可以清除所有不常用的key
