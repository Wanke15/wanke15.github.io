# pip install clickhouse-connect==0.7.0

```python
import pandas as pd
from sqlalchemy import create_engine
URL = 'impala://xxxx:xxxx'
impala_engine = create_engine(URL)

import clickhouse_connect
client = clickhouse_connect.get_client(host='xxxx', database='search', port=8123, username='', password='')


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
