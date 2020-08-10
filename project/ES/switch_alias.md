ES线上数据更新一般采用别名的方式，通过建立两个两个索引，每个时刻只有一个索引指向别名，更新数据时先更新另个没有别名的索引，在更新成功后，通过ES的别名更新API来无缝切换，做到数据更新用户无感知。

实例:
1. 建立索引
```python
import logging

from elasticsearch import Elasticsearch

logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

es_host = 'localhost'
es_port = 9092
es = Elasticsearch(hosts=es_host, port=es_port)

doc_type = "demo_data"

mappings = {
    "mappings": {
        "demo_data": {
            "properties": {
                "name": {
                    "type": "string",
                    "index": "not_analyzed"
                }
            }
        }
    }
}
all_index_names = ["alias_index_demo1", "alias_index_demo2"]

es.indices.update_aliases(index=all_index_names[0], body=mappings)
es.indices.update_aliases(index=all_index_names[1], body=mappings)
```

2. 更细数据
```python
import os
import random

def update(index_name):
    new_name = random.random.choice(["jeff", "tina", "mike", "alice"])
    _body = {"name": new_name}
    es.index(index_name, doc_type, body=_body)
```

3. 切换别名
```python
def select_index(index_alias, all_index_names):
    alias_info = es.indices.get_alias(index_alias)
    current_index = [_ for _ in alias_info.keys()]
    assert len(current_index) == 1, "More than one index point to alias: {} => {}".format(index_alias, current_index)
    current_index = current_index[0]
    ready_index = [_ for _ in all_index_names if _ != current_index][0]
    return current_index, ready_index
    
alias_name = "name_index"

current_idx, ready_idx = select_index(alias_name, all_index_names)

# recreate index
es.indices.delete(ready_idx)
es.indices.create(index=ready_idx, body=mappings)

try:
    update(ready_idx)
    logger.info("Spark job run successfully! Ready to update index alias...")
    _body = {
        "actions": [
            {"remove": {"index": current_idx, "alias": alias_name}},
            {"add": {"index": ready_idx, "alias": alias_name}}
        ]
    }
    es.indices.update_aliases(_body)

    current_idx, ready_idx = select_index(index_alias, all_index_names)
    logger.info("Update success! Index: {} in working and index: {} stands by!".format(current_idx, ready_idx))
except Exception as e:
    logger.error("Update error: {}!".format(e))
```
