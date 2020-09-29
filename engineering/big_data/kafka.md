1. Docker快速创建 => docker-compose.yml
```yml
version: '2'

services:
  zoo1:
    image: wurstmeister/zookeeper
    hostname: zoo1
    ports:
      - "2181:2181"
    container_name: zookeeper


  kafka1:
    image: wurstmeister/kafka
    ports:
      - "9092:9092"
    environment:
      KAFKA_ADVERTISED_HOST_NAME: localhost
      KAFKA_ZOOKEEPER_CONNECT: "zoo1:2181"
      KAFKA_BROKER_ID: 1
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_CREATE_TOPICS: "test_topic"
    depends_on:
      - zoo1
    container_name: kafka
```
```bash
docker-compose up -d
docker-compose down
```

2. Producer
```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    bootstrap_servers=['localhost:9092']
)
for i in range(2):
    data = {
        "name": "李四",
        "age": 23,
        "gender": "男",
        "id": i
    }
    producer.send('test_topic', data)
producer.close()
```

3. Consumer
```python
import json
from kafka import KafkaConsumer

consumer = KafkaConsumer('test_topic', bootstrap_servers='localhost:9092',
                         auto_offset_reset='latest', #
                         enable_auto_commit=True, #
                         auto_commit_interval_ms=1000, #
                         group_id='my-group', # auto_commit and group_id make sure consume one and only onece.
                         value_deserializer=json.loads)
for msg in consumer:
    print(msg.value)

```

4. SparkStreaming整合Kafka

(1) SBT依赖
```sbt
name := "SbtTest"

version := "0.1"

scalaVersion := "2.11.8"

// libraryDependencies += "com.sksamuel.elastic4s" %% "elastic4s-http" % "6.3.3"

libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.1.0"
libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.1.0"
libraryDependencies += "org.apache.spark" % "spark-hive_2.11" % "2.1.0"
libraryDependencies += "org.apache.spark" % "spark-streaming_2.11" % "2.1.0"
libraryDependencies += "org.apache.spark" % "spark-streaming-kafka-0-10_2.11" % "2.1.0"


libraryDependencies += "com.alibaba" % "fastjson" % "1.2.58"



assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

// assemblyShadeRules in assembly := Seq(
//   ShadeRule.rename("org.apache.http.**" -> "shadeio.@1").inAll
// )

```
(2) 代码
```scala
package com.streaming

import org.apache.kafka.clients.consumer.ConsumerConfig
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.kafka010.{CanCommitOffsets, ConsumerStrategies, HasOffsetRanges, KafkaUtils, LocationStrategies}
import org.apache.spark.streaming.{Seconds, StreamingContext}

object SparkStreamingKafka {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("KafkaStreaming")
      .master("local[8]")
      .getOrCreate()

    val topics = Set("test_topic")
    val params: Map[String, Object] = Map(
      ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG -> "localhost:9092",
      ConsumerConfig.GROUP_ID_CONFIG -> "test_topic_group",
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

    stream.foreachRDD(rdd => {
      val offsetRanges = rdd.asInstanceOf[HasOffsetRanges].offsetRanges

      rdd.foreachPartition(r => {
        r.foreach(str => {
          val parseData = JSON.parseObject(str.value.toString)
          val gender = parseData.get("gender")
          println(gender)
        })
      })

      stream.asInstanceOf[CanCommitOffsets].commitAsync(offsetRanges)
    })

    ssc.start()
    ssc.awaitTermination()

  }

}

```
5. Spark streaming 整合 kafka 手动提交offset

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
