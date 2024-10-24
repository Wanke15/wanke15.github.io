
### 1. pandas 交互
# pip install clickhouse-connect==0.7.0

（1）读
```sql
import clickhouse_connect
import pandas as pd

# 创建 ClickHouse 客户端
client = clickhouse_connect.get_client(host='localhost', port=8123, username='default', password='')

# 执行查询并将结果转换为 Pandas DataFrame
query = 'SELECT * FROM your_table'
result = client.query(query)
df = pd.DataFrame(result.result_rows, columns=result.column_names)
```

（2）写
```python
import pandas as pd
from sqlalchemy import create_engine
URL = 'impala://impala.idc.jianke.com:21050'
impala_engine = create_engine(URL)

import clickhouse_connect
client = clickhouse_connect.get_client(host='172.17.240.84', database='search', port=8123, username='', password='')


def compute_indicators(biz_date):
    with open("./test.sql", "r", encoding="utf8") as f:
        sql = f.read()

    sql = sql.replace("{biz_date}", biz_date)
    # print(sql)
    _df = pd.read_sql(sql, con=impala_engine)
    _df["biz_date"] = pd.to_datetime(biz_date)
    return _df


df = compute_indicators("2024-10-23")

client.insert_df('search_indicators', df)
```
