1. Spark DataFrame一行分割为多行
```scala
movies.withColumn("genre", explode(split($"genre", "[|]"))).show
+-------+---------+---------+
|movieId|movieName|    genre|
+-------+---------+---------+
|      1| example1|   action|
|      1| example1| thriller|
|      1| example1|  romance|
|      2| example2|fantastic|
|      2| example2|   action|
+-------+---------+---------+
```

2. dataframe根据某列降序
```scala
import org.apache.spark.sql.functions._
df.orderBy(desc("col2")).show

+----+----+----+
|col1|col2|col3|
+----+----+----+
|   1|   8|   6|
|   4|   5|   9|
|   7|   2|   3|
+----+----+----+
```

3. structured streaming append 和 complete
```scala
package com.streaming

import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.functions._

import com.alibaba.fastjson.JSON



object KafkaStructuredStreamingInput {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("KafkaStructuredStreaming")
      .master("local[8]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val inputDataFrame = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:9092")
      .option("subscribe", "note")
      .option("group", "structured_streaming")
      .load()

    val keyValueDataset = inputDataFrame.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)").as[(String, String)]

    val userSourceDf = keyValueDataset.map(t => {
      val parseData = JSON.parseObject(t._2)
      (parseData.getString("msg").toLowerCase())
    }).toDF( "words")


//    val result = userSourceDf.withColumn("words", explode(split($"words", "[ ]")))
//      .groupBy("words").count().orderBy(desc("count"))
//
//    val query = result.writeStream.outputMode("complete").format("console").start()

    val result = userSourceDf.withColumn("words", explode(split($"words", "[ ]")))

    val query = result.writeStream.outputMode("append").format("console").start()

    query.awaitTermination()
  }

}

```

4. kafka手动提交offset

**offset**：指的是kafka的topic中的每个**消费组**消费的下标。

offset下标自动提交在很多场景都不适用，因为自动提交是在kafka拉取到数据之后就直接提交，这样很容易丢失数据，尤其是在需要事务控制的时候。
很多情况下我们需要从kafka成功拉取数据之后，对数据进行相应的处理之后再进行提交。如拉取数据之后进行写入mysql这种 ， 所以这时我们就需要进行手动提交kafka的offset下标。
```scala
package com.streaming

import org.apache.kafka.clients.consumer.ConsumerConfig
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.TaskContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.kafka010.{CanCommitOffsets, ConsumerStrategies, HasOffsetRanges, KafkaUtils, LocationStrategies, OffsetRange}
import org.apache.spark.streaming.{Seconds, StreamingContext}

object Demo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("KafkaStreaming")
      .master("local[8]")
      .getOrCreate()

//    val topics = Set("feed_recommend_topic")
    val topics = Set("note")
    val params: Map[String, Object] = Map(
      ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG -> "localhost:9092",
      ConsumerConfig.GROUP_ID_CONFIG -> "feed_recommend",
//      ConsumerConfig.SESSION_TIMEOUT_MS_CONFIG -> 20000,
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "enable.auto.commit" -> (false: java.lang.Boolean)
    )

    val duration = 2
    val ssc = new StreamingContext(spark.sparkContext, Seconds(duration))

    val stream = KafkaUtils.createDirectStream[String, String](
      ssc,
      LocationStrategies.PreferConsistent,
      ConsumerStrategies.Subscribe[String, String](topics, params)
    )

    stream.foreachRDD(rdds => {
      val offsetRanges = rdds.asInstanceOf[HasOffsetRanges].offsetRanges
        rdds.foreachPartition(p => {
          val o: OffsetRange = offsetRanges(TaskContext.get.partitionId)
          println(s"${o.topic} ${o.partition} ${o.fromOffset} ${o.untilOffset}")
          p.foreach(str => {
            println("*******" + str.value())
          })
        })

        // 确保中间操作都完成了，再提交偏移量到kafka， 避免中间某个stage环节卡住仍自动消费提交
        stream.asInstanceOf[CanCommitOffsets].commitAsync(offsetRanges)
      })

    ssc.start()
    ssc.awaitTermination()

  }

}

```
