用户 u 和用户 v，都购买过同一件商品i ，则三者之间会构成一个类似秋千的关系图。若用户 u 和用户 v 之间除了共同购买过 i 外，还共同购买过商品 j，则认为两件商品i和j是具有某种程度上的相似的。
也就是说，商品与商品之间的相似关系，是通过用户关系来传递的。为了衡量物品 i 和 j 的相似性，考察都购买过物品 i 和 j 的用户 u 和用户 v ， 如果这两个用户共同购买的物品越少，则物品 i 和 j 的相似性越高。
相似度公式计算公式如下：

$s(i, j) = \sum_{u\in(u_i\cap u_j)} \sum_{v\in(u_i\cap u_j)} \frac{1}{\alpha + \| i_u \cap j_u \| }$

其中：
 - $u_i表示购买了i的用户集合$
 - $u_j表示购买了j的用户集合$
 - $i_u表示u购买的物品集合$
 - $i_v表示v购买的物品集合$

公式比较简单，先从数据集分别构建商品-用户map和用户-商品map并broadcast出去，然后商品之间笛卡尔积构建商品对，商品对之间应用上述公式计算相似度score

以下代码在的[Thinkgamer](https://thinkgamer.blog.csdn.net/article/details/115678598)基础上做了改进：
```scala
  
case class I2iEntity(base_id: String, rec_id: String, similarity: Double)

class SwingModel(spark: SparkSession) extends Serializable{
    var alpha: Option[Double] = Option(0.0)
    var items: Option[ArrayBuffer[String]] = Option(new ArrayBuffer[String]())
    var userIntersectionMap: Option[Map[String, Map[String, Int]]] = Option(Map[String, Map[String, Int]]())

    /*
     * @Description 给参数 alpha赋值
     * @Param double
     * @return cf.SwingModel
     **/
    def setAlpha(alpha: Double): SwingModel = {
        this.alpha = Option(alpha)
        this
    }

    /*
     * @Description 给所有的item进行赋值
     * @Param [array]
     * @return cf.SwingModel
     **/
    def setAllItems(array: Array[String]): SwingModel = {
        this.items = Option(array.toBuffer.asInstanceOf[ArrayBuffer[String]])
        this
    }

    /*
     * @Description 获取两两用户有行为的item交集个数
     * @Param [spark, data]
     * @return scala.collection.immutable.Map<java.lang.String,scala.collection.immutable.Map<java.lang.String,java.lang.Object>>
     **/
    def calUserRateItemIntersection(data: RDD[(String, String, Double)]): Map[String, Map[String, Int]] = {
        val rdd = data.map(l => (l._1, l._2)).groupByKey().map(l => (l._1, l._2.toSet))
        val map = (rdd cartesian rdd).map(l => (l._1._1, (l._2._1, (l._1._2 & l._2._2).toArray.length)))
            .groupByKey()
            .map(l => (l._1, l._2.toMap))
            .collectAsMap().toMap
        map.take(10).foreach(println)
        map
    }

    def fit(data: RDD[(String, String, Double)]): RDD[I2iEntity]= {
        this.userIntersectionMap = Option(this.calUserRateItemIntersection(data))
        println(this.userIntersectionMap.take(10))

        val rdd = data.map(l => (l._2, l._1)).groupByKey().map(l => (l._1, l._2.toSet))
        val result: RDD[I2iEntity] = (rdd cartesian rdd).map(l => {
            val item1 = l._1._1
            val item2 = l._2._1
            // 笛卡尔积会有相同的item pair，因此直接置-1，后续过滤掉这些结果
            if (item1 == item2) {
             I2iEntity(item1, item2, -1) // (item1, item2, swingsocre)
            }
            val intersectionUsers = l._1._2 & l._2._2
            var score = 0.0
            for(u1 <- intersectionUsers){
                for(u2 <- intersectionUsers){
                    score += 1.0 / (this.userIntersectionMap.get.get(u1).get(u2).toDouble + this.alpha.get)
                }
            }
            I2iEntity(item1, item2, score) // (item1, item2, swingsocre)
        })
        result
    }
}
```

```scala
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

object Swing {
    def main(args: Array[String]): Unit = {
       val spark = SparkSession.builder()
         .master("local[10]") // 集群提交时注释掉
         .config("spark.sql.session.timeZone", "Asia/Shanghai")
         .config("spark.port.maxRetries", 100)
         .config("spark.hadoop.hive.metastore.disallow.incompatible.col.type.changes", "false")
         .enableHiveSupport()
         .getOrCreate()

        val trainDataPath = "data/ml-100k/ua.base"
        val testDataPath = "data/ml-100k/ua.test"

      	import spark.sqlContext.implicits._
        val train: RDD[(String, String, Double)] = spark.sparkContext.textFile(trainDataPath).map(_.split("\t")).map(l => (l(0), l(1), l(2).toDouble))
        val test: RDD[(String, String, Double)] = spark.sparkContext.textFile(testDataPath).map(_.split("\t")).map(l => (l(0), l(1), l(2).toDouble))

        val items: Array[String] = train.map(_._2).collect()

        val swing = new SwingModel(spark).setAlpha(1).setAllItems(items)
        val itemSims: RDD[I2iEntity] = swing.fit(train)
        
        val result = itemSims.filter(e => e.similarity > 0)
        
        import spark.implicits._

        val i2iRecResult = result.toDF()

        // row_number添加相似度排名rank，并取top_k结果
        val top_k = 30
        val finalResult = i2iRecResult.select(col("base_id"), col("base_name"), col("rec_id"), col("rec_name"), col("similarity"),
          row_number().over(Window.partitionBy("base_id").orderBy(col("similarity").desc)).alias("rank")).filter($"rank" <= top_k)

        finalResult.write.mode(SaveMode.Overwrite).parquet("s3://xxx-bucket/swing_i2i)

        spark.close()
    }
}

```
