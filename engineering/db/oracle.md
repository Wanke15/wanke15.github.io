```python
import cx_Oracle
import os
os.environ["ORACLE_HOME"] = r'D:\instantclient_11_2'

user = 'xxxxx'
password = 'xxxx'
host = 'xxxxx'
port = '1521'
service_name = 'xxxxx'

# create connection
conn_str = f"{user}/{password}@{host}:{port}/{service_name}"# ('system/system@172.24.0.64:1521/helowinXDB')
connect = cx_Oracle.connect(conn_str)

import pandas as pd
from tqdm import tqdm
import numpy as np

# load data
brain_df = pd.read_csv("xxx/xxx.tsv")
brain_df.shape

# truncate
cursor = connect.cursor()
cursor.execute('TRUNCATE TABLE nsfdata.VOC_DEMO')
cursor.execute('select count(1) from nsfdata.VOC_DEMO')
print(cursor.fetchall())

# batch insert
base_sql = """INSERT INTO NSFDATA.VOC_DEMO
(CHANNEL, VOC_CONTENT, NOTE_URL,
DEMAND_SENTIMENT, DEMAND_TYPE, DEMAND_VALUE, 
NOTE_ID, NOTE_TIME, NOTE_USER_NAME, 
PROFILE_AGE, PROFILE_GENDER, PROFILE_TYPE, 
SEASON, TEXT, TITLE, 
WHAT, VOC_WHEN, VOC_WHERE, WHERE_CATE, WHO, WHY, 
COLLECTED_COUNT, COMMENTS_COUNT, DISTINCT_NOTE_COUNT, NOTE_LIKE_COUNT, SHARE_COUNT)
VALUES(:1, :2, :3, 
:4, :5, :6, 
:7, :8, :9, 
:10, :11, :12, 
:13, :14, :15,  
:16, :17, :18, :19, :20, :21,  
:22, :23, :24, :25, :26)
"""

# date format setting
cursor.execute("ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY/MM/DD'")

_insert_df = brain_df.fillna("empty").fillna(0)
batch_size = 10000
batch_list = []

cursor = connect.cursor()

for i, row in tqdm(_insert_df.iterrows(), nrows=len(brain_df)):
    batch_data = (row['Channel'], row['Content'], row["Note Url"],
                              row['Demand Sentiment'], row['Demand Type'],  row['Demand Value'], 
                              row['Note Id'], row['Note Time'],  row['Note User Name'],
                              row['Profile Age'], row['Profile Gender'],  row['Profile Type'],
                              row['Season'], row['Text'],  row['Title'],
                              row['What'], row['When'],  row['Where'], row['where_cate'], row['Who'],  row['Why'], 
                              
                              int(row['Collected Count']),
                              int(row['Comments Count']),
                              int(row['Distinct Note Count']),
                              int(row['Note Like Count']),
                              int(row['Share Count'])
                 )
    
    batch_list.append(batch_data)
    if i > 0 and i % batch_size == 0:
        cursor.executemany(base_sql, batch_list)
        connect.commit()
        batch_list = []
    
cursor.executemany(base_sql, batch_list)
connect.commit()
```
