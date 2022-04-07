# `关注点：`
1. foldLeft和foldRight操作
2. xgboost4j-spark训练时如果数据分区数和设置的num_workers数量不一致，则会先对训练数据进行重分区，因此对训练样本顺序敏感或有要求的要避免此操作
3. spark针对rdd和dataframe设置默认并行度、分区数的方法(配置)

## 1. Pairwise rank training with `groupData`

```scala

package com.learn.xgboost.train

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ListBuffer

import org.apache.spark.sql.functions.col

/**
 * @author wangke
 * @date 2022/4/7 7:54 下午
 * @version 1.0
 */
object PairWiseGroupRank {
  /**
   * 计算训练集对应的qid group
   * @param ds : 训练集
   * @param queryIdCol : qid
   * @return : 返回训练集中qid对应的长度，格式为list
   */
  def calcQueryGroups(ds: DataFrame, queryIdCol: String): Seq[Seq[Int]] = {
    val groupData = ds.select(col(queryIdCol)).rdd.mapPartitionsWithIndex {
      (partitionId, rows) =>
        rows.foldLeft(List[(String, Int)]()) {
          (acc, e) =>
            val queryId = e.getAs[String](queryIdCol)
            acc match {
              // If the head matches current queryId increment it
              case ((`queryId`, count) :: xs) => (queryId, count + 1) :: xs
              // otherwise add new item to head
              case _ => (queryId, 1) :: acc
            }
          // Replace query id with partition id
        }.map {
          case (_, count) => (partitionId, count)
          // reverse because the list was built backwards
        }.reverse.toIterator
    }.collect()

    val numPartitions = ds.rdd.getNumPartitions
    val groups = Array.fill(numPartitions)(ListBuffer[Int]())
    // Convert our list of (partitionid, count) pairs into the result
    // format. Spark guarantees to sort order has been maintained.
    for (e <- groupData) {
      groups(e._1) += e._2
    }
    groups
  }

  def main(args: Array[String]): Unit = {
    // 默认并行度和分区数设置为workers数量，避免fit时重分区，打乱了groupData的顺序
    val numWorkers = 4

    val spark = SparkSession.builder()
      .appName("xgb")
      .master("local[*]")
      .config("spark.sql.session.timeZone", "Asia/Shanghai")
      .config("spark.default.parallelism", numWorkers) // rdd默认并行度
      .config("spark.sql.shuffle.partitions", numWorkers) // dataframe默认分区数
      .getOrCreate()
      
    // EMR集群
//    val spark = SparkSession.builder()
//      .config("spark.sql.session.timeZone", "Asia/Shanghai")
//      .config("spark.port.maxRetries", 100)
//      .config("spark.hadoop.hive.metastore.disallow.incompatible.col.type.changes", "false")
//      .enableHiveSupport()
//      .getOrCreate()

    val rawData: DataFrame = spark.read.option("inferSchema", value = true)
      .option("header", value = true)
      .csv("src/main/resources/data/iris-two-group.csv")

    val groupCol = "query_id"

    // 根据groupCol排序，保证同一个group的数据连续
    val data = rawData.sort(groupCol)

    data.show(false)
    data.printSchema()
    
    val vectorAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array("x1", "x2", "x3", "x4")).setOutputCol("features")
    val train: DataFrame = vectorAssembler.transform(data)

    // ------ get group id :
    val queryCount = train.groupBy(groupCol).count().select(groupCol, "count")
      .withColumnRenamed(groupCol, s"${groupCol}_new")

    queryCount.show(false)


    val trainWithGroupID = train.join(queryCount, col(groupCol) === col(s"${groupCol}_new"), "left_outer")

    trainWithGroupID.show(false)

    // ------ set group id :
    val groupQueryID = calcQueryGroups(trainWithGroupID, groupCol) // Seq(Seq(Int))

    groupQueryID.foreach(println)

    val paramsMap = Map(
      ("eta" -> 0.1f),
      ("max_depth" -> 3),
      ("objective" -> "rank:pairwise"),
      ("num_round" -> 3),
      ("num_workers" -> numWorkers),
      ("eval_metric" -> "logloss"),
      ("groupData", groupQueryID)
    )

    println("fit前的分区数：" + trainWithGroupID.rdd.getNumPartitions)

    val xgb = new XGBoostClassifier(xgboostParams = paramsMap)
    xgb.setFeaturesCol("features")
    xgb.setLabelCol("label")

    val clf: XGBoostClassificationModel = xgb.fit(trainWithGroupID)

    println("fit后的分区数：" + trainWithGroupID.rdd.getNumPartitions)


    val trainPrediction: DataFrame = clf.transform(trainWithGroupID)
    trainPrediction.show(100, false)


  }

}

```

## 2. iris-two-group.csv
```csv
query_id,x1,x2,x3,x4,label
q1,5.1,3.5,1.4,0.2,0
q1,4.9,3.0,1.4,0.2,0
q1,4.7,3.2,1.3,0.2,0
q1,4.6,3.1,1.5,0.2,0
q2,5.0,3.6,1.4,0.2,0
q2,5.4,3.9,1.7,0.4,0
q2,4.6,3.4,1.4,0.3,0
q2,5.0,3.4,1.5,0.2,0
q2,4.4,2.9,1.4,0.2,0
q2,4.9,3.1,1.5,0.1,0
q2,5.4,3.7,1.5,0.2,0
q2,4.8,3.4,1.6,0.2,0
q2,4.8,3.0,1.4,0.1,0
q2,4.3,3.0,1.1,0.1,0
q3,5.8,4.0,1.2,0.2,0
q3,5.7,4.4,1.5,0.4,0
q3,5.4,3.9,1.3,0.4,0
q3,5.1,3.5,1.4,0.3,0
q3,5.7,3.8,1.7,0.3,0
q3,5.1,3.8,1.5,0.3,0
q3,5.4,3.4,1.7,0.2,0
q3,5.1,3.7,1.5,0.4,0
q4,4.6,3.6,1.0,0.2,0
q4,5.1,3.3,1.7,0.5,0
q4,4.8,3.4,1.9,0.2,0
q5,5.0,3.0,1.6,0.2,0
q5,5.0,3.4,1.6,0.4,0
q5,5.2,3.5,1.5,0.2,0
q5,5.2,3.4,1.4,0.2,0
q5,4.7,3.2,1.6,0.2,0
q5,4.8,3.1,1.6,0.2,0
q5,5.4,3.4,1.5,0.4,0
q5,5.2,4.1,1.5,0.1,0
q5,5.5,4.2,1.4,0.2,0
q5,4.9,3.1,1.5,0.2,0
q5,5.0,3.2,1.2,0.2,0
q5,5.5,3.5,1.3,0.2,0
q5,4.9,3.6,1.4,0.1,0
q5,4.4,3.0,1.3,0.2,0
q5,5.1,3.4,1.5,0.2,0
q5,5.0,3.5,1.3,0.3,0
q5,4.5,2.3,1.3,0.3,0
q5,4.4,3.2,1.3,0.2,0
q5,5.0,3.5,1.6,0.6,0
q5,5.1,3.8,1.9,0.4,0
q5,4.8,3.0,1.4,0.3,0
q5,5.1,3.8,1.6,0.2,0
q6,4.6,3.2,1.4,0.2,0
q6,5.3,3.7,1.5,0.2,0
q6,5.0,3.3,1.4,0.2,0
q1,7.0,3.2,4.7,1.4,1
q1,6.4,3.2,4.5,1.5,1
q2,6.9,3.1,4.9,1.5,1
q2,5.5,2.3,4.0,1.3,1
q2,6.5,2.8,4.6,1.5,1
q2,5.7,2.8,4.5,1.3,1
q2,6.3,3.3,4.7,1.6,1
q2,4.9,2.4,3.3,1.0,1
q2,6.6,2.9,4.6,1.3,1
q2,5.2,2.7,3.9,1.4,1
q3,5.0,2.0,3.5,1.0,1
q3,5.9,3.0,4.2,1.5,1
q3,6.0,2.2,4.0,1.0,1
q3,6.1,2.9,4.7,1.4,1
q3,5.6,2.9,3.6,1.3,1
q3,6.7,3.1,4.4,1.4,1
q3,5.6,3.0,4.5,1.5,1
q3,5.8,2.7,4.1,1.0,1
q3,6.2,2.2,4.5,1.5,1
q3,5.6,2.5,3.9,1.1,1
q4,5.9,3.2,4.8,1.8,1
q4,6.1,2.8,4.0,1.3,1
q4,6.3,2.5,4.9,1.5,1
q4,6.1,2.8,4.7,1.2,1
q4,6.4,2.9,4.3,1.3,1
q4,6.6,3.0,4.4,1.4,1
q4,6.8,2.8,4.8,1.4,1
q4,6.7,3.0,5.0,1.7,1
q4,6.0,2.9,4.5,1.5,1
q4,5.7,2.6,3.5,1.0,1
q5,5.5,2.4,3.8,1.1,1
q5,5.5,2.4,3.7,1.0,1
q5,5.8,2.7,3.9,1.2,1
q5,6.0,2.7,5.1,1.6,1
q5,5.4,3.0,4.5,1.5,1
q5,6.0,3.4,4.5,1.6,1
q6,6.7,3.1,4.7,1.5,1
q6,6.3,2.3,4.4,1.3,1
q6,5.6,3.0,4.1,1.3,1
q6,5.5,2.5,4.0,1.3,1
q6,5.5,2.6,4.4,1.2,1
q6,6.1,3.0,4.6,1.4,1
q6,5.8,2.6,4.0,1.2,1
q6,5.0,2.3,3.3,1.0,1
q6,5.6,2.7,4.2,1.3,1
q6,5.7,3.0,4.2,1.2,1
q6,5.7,2.9,4.2,1.3,1
q6,6.2,2.9,4.3,1.3,1
q6,5.1,2.5,3.0,1.1,1
q6,5.7,2.8,4.1,1.3,1

```
