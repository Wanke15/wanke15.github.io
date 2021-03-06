Redis, MongoDB, ElasticSearch, HBase

1. 如果你对数据的**读写要求极高**，并且你的**数据规模不大**，也不需要长期存储，选**redis**
2. 如果你的数据规模较大，对数据的**读性能**要求很高，**数据表的结构需要经常变**，有时还需要做一些聚合查询，选**MongoDB**
3. 如果你需要构造一个**搜索引擎**或者你想搞一个看着高大上的数据可视化平台，并且你的数据有一定的分析价值，选**ElasticSearch**
4. 如果你需要存储**海量数据**，连你自己都不知道你的数据规模将来会增长多么大，那么选**HBase**

<img src="https://raw.githubusercontent.com/Wanke15/wanke15.github.io/master/engineering/assets/nosql-db.jpeg">
