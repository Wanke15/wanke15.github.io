```sql
-- Kafka行为消息source
CREATE TABLE user_behavior (
  `who` STRING,
  `behavior_type` STRING,
  `store_product_id` STRING,
  proctime AS PROCTIME(),
   ts as TO_TIMESTAMP(FROM_UNIXTIME(`when`['time_rep']/1000,'yyyy-MM-dd HH:mm:ss')),
  WATERMARK FOR ts AS ts - INTERVAL '60' MINUTE
)  with (
  'connector' = 'kafka',
  'properties.group.id' = 'test',
  'properties.bootstrap.servers' = 'xxxx:9092',
  'properties.key.serializer' = 'org.apache.kafka.common.serialization.ByteArraySerializer',
  'properties.value.serializer' = 'org.apache.kafka.common.serialization.ByteArraySerializer',
  'topic' = 'user_behavior_sink',
  'scan.startup.mode' = 'group-offsets',
  'format' = 'avro'
);

-- 行为权重映射
create temporary view user_behavior_search as
SELECT `who` as user_id, store_product_id, 
case when behavior_type = '1' then 0.8 -- 提醒
when behavior_type = '2' then 0.4 -- 加购
when behavior_type = '3' then 0.2 -- 点击
else 0.0 end as behavior_weight, proctime FROM user_behavior where behavior_type in ('1', '2', '3') and `who` is not null and trim(`who`) <> '';

-- 行为权重聚合
create temporary view user_store_product_weight as
SELECT user_id, store_product_id, sum(behavior_weight) as score
FROM user_behavior_search
GROUP BY HOP(proctime, INTERVAL '5' SECOND, INTERVAL '10' MINUTE), user_id, store_product_id;

-- topN喜好商品
create temporary view user_store_product_topn as
SELECT user_id, store_product_id, score, row_num
FROM 
(
   SELECT user_id, store_product_id, score, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score desc) AS row_num
   FROM user_store_product_weight
)
WHERE row_num <= 10;

-- 结果sink
create table ub_storeproduct_print (
  `src`    string,
  `dst`    string, 
  `score`   double,
  `ts`   int
) with (
  'connector'='print'
);

begin statement set;

insert into ub_storeproduct_print select `user_id`, `store_product_id`,  `score`, cast(current_timestamp as int) from user_store_product_topn;

end;
```
