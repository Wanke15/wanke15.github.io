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

## 3. array 类型数据批量导入
由于方法2并不兼容array类型数据导入，在尝试了一些方法后，最终采用如下的批量导入方式
```
# 导入数据
import pymysql

conn = pymysql.connect(host='xxx', port=9030, user='xxx', password='xxx', charset='utf8', db='xxx')
cur = conn.cursor()

columns = new_df.columns
col_len = len(columns)
record_num = len(new_df)

table_name = 'xxx.xxx'
batch_size = 20000

base_sql = """INSERT INTO {} {}
         values {};
    """.format(table_name, "(" + ",".join(columns) + ")", "(" + ",".join(['%s' for _ in range(len(columns))]) + ")")
print(base_sql)

data_list = []
for i, row in tqdm(new_df.iterrows(), total=record_num):
    record = ["{}".format(row[c]) for c in columns]
    data_list.append(record)

    if i % batch_size == 0 and i > 1 or i == record_num:
        print("批量提交： ", i)
        cur.executemany(base_sql, data_list)
        conn.commit()
        data_list = []
```
