```python
import pickle

from elasticsearch import helpers, Elasticsearch
from tqdm import tqdm

es = Elasticsearch(hosts='http://localhost', port=9200)


_body = {
    "query": {
        "bool": {
            "must": [
                {
                    "match_all": {}
                }
            ],
            "should": []
        }
    },
    "from": 0,
    "size": 10000
}
scan_resp = helpers.scan(es, _body, scroll="10m", index="data_index", doc_type="data_type", timeout="10m")

all_res = [resp['_source'] for resp in tqdm(scan_resp)]

print('Saving...')
with open('result.pkl', 'wb') as f:
    pickle.dump(all_res, f)
  
```
