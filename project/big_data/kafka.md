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
        rdd.foreachPartition(p => {
          println(p.mkString(", "))
        })
        val offsetRanges = rdd.asInstanceOf[HasOffsetRanges].offsetRanges
        stream.asInstanceOf[CanCommitOffsets].commitAsync(offsetRanges)
      })

    ssc.start()
    ssc.awaitTermination()

  }

}

```
