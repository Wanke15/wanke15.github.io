以15分钟为例：

### 1. 基于Hive
```sql
INSERT INTO TABLE user_behavior_sessions
WITH user_behavior_sorted AS (
    SELECT
        user_id,
        item_id,
        behavior_type,
        behavior_time,
        CAST(UNIX_TIMESTAMP(behavior_time, 'yyyy-MM-dd HH:mm:ss') AS BIGINT) AS behavior_timestamp
    FROM user_behavior
    ORDER BY user_id, behavior_time
),
user_behavior_diff AS (
    SELECT
        user_id,
        item_id,
        behavior_type,
        behavior_time,
        behavior_timestamp,
        LAG(behavior_timestamp, 1) OVER (PARTITION BY user_id ORDER BY behavior_time) AS prev_behavior_timestamp
    FROM user_behavior_sorted
),
user_behavior_session AS (
    SELECT
        user_id,
        item_id,
        behavior_type,
        behavior_time,
        behavior_timestamp,
        prev_behavior_timestamp,
        CASE
            WHEN prev_behavior_timestamp IS NULL OR (behavior_timestamp - prev_behavior_timestamp) > 900 THEN 1
            ELSE 0
        END AS new_session_flag
    FROM user_behavior_diff
),
user_behavior_with_session_id AS (
    SELECT
        user_id,
        item_id,
        behavior_type,
        behavior_time,
        behavior_timestamp,
        prev_behavior_timestamp,
        new_session_flag,
        SUM(new_session_flag) OVER (PARTITION BY user_id ORDER BY behavior_time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_id
    FROM user_behavior_session
)
SELECT
    user_id,
    item_id,
    behavior_type,
    behavior_time,
    session_id
FROM user_behavior_with_session_id;
```

### 2. 基于Pandas dataframe
```python
import pandas as pd

# 假设df是你的数据框
# 示例数据框
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3],
    'item_id': [101, 102, 103, 201, 202, 203, 301],
    'behavior_type': ['click', 'click', 'buy', 'click', 'click', 'buy', 'click'],
    'behavior_time': pd.to_datetime([
        '2024-09-11 10:00:00', 
        '2024-09-11 10:10:00', 
        '2024-09-11 10:30:00', 
        '2024-09-11 11:00:00', 
        '2024-09-11 11:05:00', 
        '2024-09-11 11:20:00', 
        '2024-09-11 12:00:00'
    ])
}
df = pd.DataFrame(data)

# 按user_id和behavior_time排序
df = df.sort_values(by=['user_id', 'behavior_time'])

# 计算时间差
df['prev_behavior_time'] = df.groupby('user_id')['behavior_time'].shift(1)
df['time_diff'] = (df['behavior_time'] - df['prev_behavior_time']).dt.total_seconds()

# 标记新的session
df['new_session'] = (df['time_diff'] > 900) | (df['time_diff'].isna())

# 生成session_id
df['session_id'] = df.groupby('user_id')['new_session'].cumsum()

# 删除多余列
df = df.drop(columns=['prev_behavior_time', 'time_diff', 'new_session'])

print(df)
```

