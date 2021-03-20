### 基于搜索加购日志计算物品共现相似度(Jaccard similarity)

```sql
-- query_log 为搜索日志宽表；得到加购某商品使用的所有query
create or replace temporary view product_query_uniq as
select product_id, product_name, explode(queries) as query from (
select product_id, product_name, collect_set(q_query) as queries from query_log
where product_id is not null and q_query != '' and q_query is not null
and add_cart = 1
and cur_time >= concat_ws('', split(date_sub(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'), '${days}'), '-'))
group by product_id, product_name
);

-- 全量相似度临时表
create or replace temporary view i2i_tbl as 
select base_id, base_name, rec_id, rec_name, row_number() over (partition by id1 ORDER BY similarity desc) rank from (
SELECT t1.product_id as base_id, t1.product_name as base_name,
       t2.product_id as rec_id, t2.product_name as rec_name,
       1.0*sum(CASE WHEN t1.query = t2.query THEN 1 ELSE 0 END)
     / (count(DISTINCT t1.query)+count(DISTINCT t2.query)-sum(CASE WHEN t1.query = t2.query THEN 1 ELSE 0 END)) AS similarity
FROM product_query_uniq t1
JOIN product_query_uniq t2 ON t1.product_id != t2.product_id
GROUP BY t1.product_id, t2.product_id, t1.product_name, t2.product_name
having similarity > 0.0
);

-- 截取top50
INSERT OVERWRITE DIRECTORY '${output_path}cur_time=${year}${month}${day}' USING PARQUET
select base_id, base_name, rec_id, rec_name, similarity, rank from i2i_tbl where rank <= 50;

```
