## 1. 安装
```bash
docker run -p 9030:9030 -p 8043:8030 -p 8041:8040 --name starRocks -itd starrocks/allin1-ubuntu:latest
docker exec -it starRocks bash
mysql -h127.0.0.1 -uroot -P9030
```

## 2. python数据导入
```python
# sqlalchemy.__version__, pandas.__version__, pymysql.__version__   =>   ('1.4.46', '1.3.5', '1.4.6')

from sqlalchemy import create_engine
 
connect_info = 'mysql+pymysql://root:@__ip__:9030/__db_name__?charset=utf8'
engine = create_engine(connect_info)

brain_df["NOTE_ID"] = brain_df["Note Id"]
brain_df["VOC_WHERE"] = brain_df["Where"]

test_small_df = brain_df[['NOTE_ID', 'VOC_WHERE', 'where_cate']]

test_small_df.to_sql(name = 'VOC_DEMO_TEST',
           con = engine, 
           chunksize=10000,
           if_exists = 'replace',
           index = False,
           )
```
