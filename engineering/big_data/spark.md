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
4. spark streaming 程序的几点优化
 - 合理的批处理时间，即 **batchDuration** 参数。一方面，如果时间过短，会造成数据的堆积，即未完成的batch数据越来越多；另一方面，如果时间过长，会造成数据延迟较大同时，也会影像整个系统的吞吐量。
   那么该如何合理的设置该参数呢？需要根据应用的实时性要求、集群的资源情况，以及通过观察spark streaming 的运行情况，尤其是Total Delay来合理的设置该参数，如下所示，最开始我在自己电脑上离线开发尝试的batchDurationw为两秒，基本满足本地开发的时延，提交到集群后，因为网络和计算资源比本地开发好很多，此时观察的Total Delay的平均值为71ms，因此可以把之前的batchDuration设置的再小一点
   <img src=assets/total-delay.png>
   
 - 合理的kafka拉取量，即 **maxRatePerPartition** 参数。当我们的数据源是kafaka时，默认的maxRatePerPartition是无上限的，即Kafka有多少数据，spark streaming 就会一次性全拉出来，如果Kafka数据更新频率很高，但是spark streaming 的批处理时间是一定的，不可能动态变化，因此此时就会造成数据堆积，阻塞的情况。所以需要结合batchDuration的值，来调整maxRatePerPartition，注意一点是数据总量是 partitionNum * maxRatePerPartition, 可以通过观察 inputRate和Processing Time来合理的设置这两个参数，使得数据的拉取和处理能够平衡
 <img src=assets/input-rate.png>
 <img src=assets/processing-time.png>
