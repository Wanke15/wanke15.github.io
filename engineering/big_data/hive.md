#### 1. 有序collect_set
```sql
select user_id, sort_array(collect_set(concat_ws(",", cast(pickup_datetime as string), cast(pickup_country_id AS string)))) 
from fact_order 
group by user_id 
limit 100a
```

#### 2. 统计用户最喜欢的Top_n个物品类型
```sql
select muid, GROUP_CONCAT(concat_ws(':', source_id, cast(source_count as string)))
 from (
        select muid, source_id, source_count, ROW_NUMBER() OVER (PARTITION BY muid ORDER BY source_count desc) rn
        from (
            select muid, source_id, count(*) as source_count
            from feed_click_event 
            where source_id is not null
            group by muid, source_id
            having source_count >= 3 # 至少有过3次点击
        ) as t
    ) as t2
where rn <= 3 # top_n
group by muid
order by muid desc
```

#### 3. 用户画像表

 - 创建表
```sql
create table common.user_profile 
(
    muid string COMMENT 'muid'
)

ROW FORMAT DELIMITED FIELDS TERMINATED BY ','

STORED AS PARQUET;
```
- 添加字段
```sql
alter table label_system_db.user_profile add columns (user_id string)
```
- 修改字段
```sql
alter table label_system_db.user_profile change column user_id user_id string comment 'user_id'
```
