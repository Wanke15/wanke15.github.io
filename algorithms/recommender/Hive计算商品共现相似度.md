### 1. 基于搜索加购日志计算物品共现相似度(Jaccard similarity)

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

### 2. Spark性能改进版
```scala
val product_query_df = spark.sql("""select product_id, product_name, collect_set(query) as queries from query_log
where product_id is not null and query != '' and query is not null
and add_cart = 1 and log_type = 2
and cur_time >= %s
group by product_id, product_name""".format(startDate))

product_query_df.cache()

// 商品之间笛卡尔积，构建商品对
val product_cartesian = product_query_df.rdd.cartesian(product_query_df.rdd)

// 计算相似度
val i2iRecResultRdd = product_cartesian.map(rdd => {
  val base_product = rdd._1
  val rec_product = rdd._2

  val base_id = base_product.getString(0)
  val base_name = base_product.getString(1)

  val rec_id = rec_product.getString(0)
  val rec_name = rec_product.getString(1)

  if (base_id == rec_id) {
    I2iEntity(base_id, base_name, rec_id, rec_name, -1)
  } else {
    val base_qs = base_product.getSeq[String](2).toSet
    val rec_qs = rec_product.getSeq[String](2).toSet

    val intersect_set = base_qs.intersect(rec_qs)
    val union_len = base_qs.size + rec_qs.size - intersect_set.size

    val sim = intersect_set.size * 1.0 / union_len
    I2iEntity(base_id, base_name, rec_id, rec_name, sim)
  }
}).filter(e => e.similarity > 0)

import spark.implicits._

val i2iRecResult = i2iRecResultRdd.toDF()

// row_number添加相似度排名rank
val finalResult = i2iRecResult.select(col("base_id"), col("base_name"), col("rec_id"), col("rec_name"), col("similarity"),
  row_number().over(Window.partitionBy("base_id").orderBy(col("similarity").desc)).alias("rank")).filter($"rank" <= top_k)

val final_output_path = "%scur_time=%s%s%s".format(output_path, year, month, day)

finalResult.write.mode(SaveMode.Overwrite).parquet(final_output_path)
```
