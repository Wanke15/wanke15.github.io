##### 1. 多线程更新(用于用户画像数据更新)
```python
import time
from concurrent.futures.thread import ThreadPoolExecutor

from impala.dbapi import connect
import pymongo

conn = connect(host='x.x.x.x', port=21050)
cursor = conn.cursor()

client = pymongo.MongoClient("x.x.x.x", 27017, username='xxx', password='xxx')
db = client.user_profile


def multi_thread_update(records):
    print("Start update...")
    start_time = time.time()
    pool = ThreadPoolExecutor()
    for res in records:
        muid_info = res.pop("muid")
        pool.submit(db.user_profile.update_one, {"muid": muid_info}, {"$set": res}, True)
        # db.user_profile.update_one({"muid": muid_info}, {"$set": res}, True)

    pool.shutdown(wait=True)
    end_time = time.time()
    print("Done, time consumed: {:.3f} seconds".format(end_time - start_time))

```
##### 2. 根据数组长度查询
```json
{ "$where": "this.favorite_feed_types.length > 1"}
{ "$where": "this.favorite_feed_types.length == 2"}
```
