```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from elasticsearch import Elasticsearch

es = Elasticsearch(hosts='http://127.0.0.1', port=9200)
mappings = {
    "mappings": {
        "_doc": {
            "properties": {
                "base_name": {
                    "type": "keyword"
                },
                "base_id": {
                    "type": "keyword"
                },
                "rank": {
                    "type": "long"
                },
                "similarity": {
                    "type": "double"
                },
                "cur_time": {
                    "type": "keyword"
                },
                "rec_name": {
                    "type": "keyword"
                },
                "rec_id": {
                    "type": "keyword"
                }
            }
        }
    }
}
index_name = "recommend_i2i"
if not es.indices.exists(index_name):
    es.indices.create(index=index_name, body=mappings)
else:
    es.indices.put_mapping(index=index_name, doc_type="_doc", body=mappings["mappings"]["_doc"])

```
