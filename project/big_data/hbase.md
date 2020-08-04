1. 创建用户画像表
```sh
create 'feed_user_profile', 'basic', 'rent_car', 'feed'
```
create 之后依次为: 表名 -> 'feed_user_profile', 三个列族 -> 'basic', 'rent_car', 'feed'

2. 写入数据
```sh
put 'feed_user_profile', 'uhdiaoe8cbb00j-ijc_2020012', 'basic:age', '20'
```
put 之后依次为： 表名 -> 'feed_user_profile', row_key -> 'uhdiaoe8cbb00j-ijc_2020012', basic列族下age字段设置为20

3. Spark读写HBase
```java
package com.feed.data

import org.apache.hadoop.hbase.client.{HTable,Put}
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapred.TableOutputFormat
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.mapred.JobConf
import org.apache.log4j.{Level,Logger}
import org.apache.spark.sql.SparkSession

case class UserProfile(sex:String, age:Int)

object HbaseMain {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val spark = SparkSession
      .builder()
      .appName("FeedDataClean")
      .master("local[8]")
      .config("hive.metastore.uris", "thrift://x.x.x.x:9083")
      .config("hive.metastore.warehouse.dir", "/user/hive/warehouse")
      .config("spark.debug.maxToStringFields", "100")
      .enableHiveSupport()
      .getOrCreate()


    val tableName = "feed_user_profile"
    val quorum = "x.x.x.x"
    val port = "2181"

    // 配置相关信息
    val conf = HBaseConfiguration.create()
    conf.set("hbase.zookeeper.quorum",quorum)
    conf.set("hbase.zookeeper.property.clientPort",port)
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") // 用KryoSerializer优化spark的序列化
    conf.set(TableInputFormat.INPUT_TABLE, tableName)

    // HBase数据转成RDD
    val hBaseRDD = spark.sparkContext.newAPIHadoopRDD(conf,classOf[TableInputFormat],
      classOf[org.apache.hadoop.hbase.io.ImmutableBytesWritable],
      classOf[org.apache.hadoop.hbase.client.Result]).cache()

    import spark.implicits._
    
    val resultDf = hBaseRDD.map(result => {
      val sex_filed= Bytes.toString(result._2.getValue("basic".getBytes(), "sex".getBytes()))
      val age_filed= Bytes.toString(result._2.getValue("basic".getBytes(), "age".getBytes())).toInt
      UserProfile(sex_filed, age_filed)
    }).toDF("sex", "age")

    resultDf.show(false)

    val dataRDD = spark.sparkContext.makeRDD(Array("00000000-595f-4b72-ffff-ffff8bf768fe_,1,16"))

    val data = dataRDD.map{ item =>
      val Array(key, sex, age) = item.split(",")
      val rowKey = key
      val put = new Put(Bytes.toBytes(rowKey))
      /*一个Put对象就是一行记录，在构造方法中指定主键
       * 所有插入的数据 须用 org.apache.hadoop.hbase.util.Bytes.toBytes 转换
       * Put.addColumn 方法接收三个参数：列族，列名，数据*/
      put.addColumn(Bytes.toBytes("basic"), Bytes.toBytes("sex"), Bytes.toBytes(sex))
      put.addColumn(Bytes.toBytes("basic"), Bytes.toBytes("age"), Bytes.toBytes(age))
      (new ImmutableBytesWritable(), put)
    }

    //保存到HBase表
    val jobConf = new JobConf(conf)
    jobConf.setOutputFormat(classOf[TableOutputFormat])
    jobConf.set(TableOutputFormat.OUTPUT_TABLE, tableName)
    data.saveAsHadoopDataset(jobConf)
  }
}
```
