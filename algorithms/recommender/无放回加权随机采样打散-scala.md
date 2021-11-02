```scala
package probability

import breeze.linalg.DenseVector
import breeze.plot.{Figure, plot}

import scala.util.Random
import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag

/**
 * @author wangke
 * @date 2021/11/2 10:47 上午
 * @version 1.0
 */

case class RecommendItem(product_id:String, product_name:String, recall_type:String, rank:Double) extends Serializable

object WeightedSampling {
  def weightedSampleWithReplacement[T:ClassTag](data: Array[T],
                                                weights: Array[Double],
                                                n: Int,
                                                random: Random): Array[T] = {
    val cumWeights = weights.scanLeft(0.0)(_ + _)
    val cumProbs = cumWeights.map(_ / cumWeights.last)
    Array.fill(n) {
      val r = random.nextDouble()
      data(cumProbs.indexWhere(r < _) - 1)
    }
  }

  /**
   * 根据权重随机采样，得到数据位置索引
   * @param weights
   * @param random
   * @return
   */
  def weightedSample(
                     weights: Seq[Double],
                     random: Random): Int = {
    val cumWeights = weights.scanLeft(0.0)(_ + _)
    val cumProbs = cumWeights.map(_ / cumWeights.last)
    val r = random.nextDouble()
    cumProbs.indexWhere(r < _) - 1
  }

  /**
   * 根据推荐结果rank值无放回加权随机采样
   * @param data
   * @return
   */
  def weightedSampleWithoutReplacement(data: Seq[RecommendItem]): Seq[RecommendItem] = {
    val dataListBuffer = new ListBuffer[RecommendItem]
    data.foreach(dataListBuffer += _)
    val result = new ListBuffer[RecommendItem]
    val rand = Random

    while (dataListBuffer.nonEmpty) {
      val weights = dataListBuffer.map(_.rank)
      val sampleIndex = weightedSample(weights, rand)
      result += dataListBuffer(sampleIndex)
      dataListBuffer.remove(sampleIndex)
    }
    result
  }

  def drawSampleWeight(weightsSeq:Seq[Array[Double]]): Unit = {
    val f = Figure()
    val p = f.subplot(0)
    weightsSeq.foreach(weights => {
      val inds:Array[Double] = weights.indices.toArray.map(_.toDouble)
      p += plot(DenseVector(inds), DenseVector(weights))
    })

    p.xlabel = "Old rank"
    p.ylabel = "New rank"
    p.title = "Weighted Sample Vis"
    f.saveas("weightedSampleVis.png")
  }

  def drawSampleWeightSingleRecord(weights:Array[Double]): Unit = {
    val f = Figure()
    val p = f.subplot(0)
    val inds:Array[Double] = weights.indices.toArray.map(_.toDouble)
    p += plot(DenseVector(inds), DenseVector(weights))

    p.xlabel = "Old rank"
    p.ylabel = "New rank"
    p.title = "Weighted Sample Vis"
    f.saveas("weightedSampleVis.png")
  }

  def main(args: Array[String]): Unit = {
    val recs = ListBuffer[RecommendItem]()
    val maxRank = 60
    for (i <- 0 until maxRank) {
      val rec = RecommendItem(s"p${i}", s"pn${i}", "", maxRank - i)
      recs += rec
    }

    println(recs)

    println(weightedSampleWithoutReplacement(recs))

//    var sampleResultList = ListBuffer[Array[Double]]()
//    for (_ <- 0 until 10) {
//       sampleResultList += weightedSampleWithoutReplacement(recs).map(_.rank).toArray
//    }
//    drawSampleWeight(sampleResultList)

//    drawSampleWeightSingleRecord(weightedSampleWithoutReplacement(recs).map(_.rank).toArray)

    for (_ <- 0 until 10) {
      drawSampleWeightSingleRecord(weightedSampleWithoutReplacement(recs).map(_.rank).toArray)
      Thread.sleep(5000)
    }

  }
}

```

```bash
ListBuffer(RecommendItem(p0,pn0,,60.0), RecommendItem(p1,pn1,,59.0), RecommendItem(p2,pn2,,58.0), RecommendItem(p3,pn3,,57.0), RecommendItem(p4,pn4,,56.0), RecommendItem(p5,pn5,,55.0), RecommendItem(p6,pn6,,54.0), RecommendItem(p7,pn7,,53.0), RecommendItem(p8,pn8,,52.0), RecommendItem(p9,pn9,,51.0), RecommendItem(p10,pn10,,50.0), RecommendItem(p11,pn11,,49.0), RecommendItem(p12,pn12,,48.0), RecommendItem(p13,pn13,,47.0), RecommendItem(p14,pn14,,46.0), RecommendItem(p15,pn15,,45.0), RecommendItem(p16,pn16,,44.0), RecommendItem(p17,pn17,,43.0), RecommendItem(p18,pn18,,42.0), RecommendItem(p19,pn19,,41.0), RecommendItem(p20,pn20,,40.0), RecommendItem(p21,pn21,,39.0), RecommendItem(p22,pn22,,38.0), RecommendItem(p23,pn23,,37.0), RecommendItem(p24,pn24,,36.0), RecommendItem(p25,pn25,,35.0), RecommendItem(p26,pn26,,34.0), RecommendItem(p27,pn27,,33.0), RecommendItem(p28,pn28,,32.0), RecommendItem(p29,pn29,,31.0), RecommendItem(p30,pn30,,30.0), RecommendItem(p31,pn31,,29.0), RecommendItem(p32,pn32,,28.0), RecommendItem(p33,pn33,,27.0), RecommendItem(p34,pn34,,26.0), RecommendItem(p35,pn35,,25.0), RecommendItem(p36,pn36,,24.0), RecommendItem(p37,pn37,,23.0), RecommendItem(p38,pn38,,22.0), RecommendItem(p39,pn39,,21.0), RecommendItem(p40,pn40,,20.0), RecommendItem(p41,pn41,,19.0), RecommendItem(p42,pn42,,18.0), RecommendItem(p43,pn43,,17.0), RecommendItem(p44,pn44,,16.0), RecommendItem(p45,pn45,,15.0), RecommendItem(p46,pn46,,14.0), RecommendItem(p47,pn47,,13.0), RecommendItem(p48,pn48,,12.0), RecommendItem(p49,pn49,,11.0), RecommendItem(p50,pn50,,10.0), RecommendItem(p51,pn51,,9.0), RecommendItem(p52,pn52,,8.0), RecommendItem(p53,pn53,,7.0), RecommendItem(p54,pn54,,6.0), RecommendItem(p55,pn55,,5.0), RecommendItem(p56,pn56,,4.0), RecommendItem(p57,pn57,,3.0), RecommendItem(p58,pn58,,2.0), RecommendItem(p59,pn59,,1.0))
ListBuffer(RecommendItem(p2,pn2,,58.0), RecommendItem(p36,pn36,,24.0), RecommendItem(p0,pn0,,60.0), RecommendItem(p7,pn7,,53.0), RecommendItem(p37,pn37,,23.0), RecommendItem(p10,pn10,,50.0), RecommendItem(p21,pn21,,39.0), RecommendItem(p16,pn16,,44.0), RecommendItem(p1,pn1,,59.0), RecommendItem(p15,pn15,,45.0), RecommendItem(p47,pn47,,13.0), RecommendItem(p23,pn23,,37.0), RecommendItem(p41,pn41,,19.0), RecommendItem(p34,pn34,,26.0), RecommendItem(p14,pn14,,46.0), RecommendItem(p44,pn44,,16.0), RecommendItem(p31,pn31,,29.0), RecommendItem(p6,pn6,,54.0), RecommendItem(p58,pn58,,2.0), RecommendItem(p25,pn25,,35.0), RecommendItem(p3,pn3,,57.0), RecommendItem(p4,pn4,,56.0), RecommendItem(p20,pn20,,40.0), RecommendItem(p12,pn12,,48.0), RecommendItem(p28,pn28,,32.0), RecommendItem(p8,pn8,,52.0), RecommendItem(p39,pn39,,21.0), RecommendItem(p49,pn49,,11.0), RecommendItem(p18,pn18,,42.0), RecommendItem(p46,pn46,,14.0), RecommendItem(p35,pn35,,25.0), RecommendItem(p11,pn11,,49.0), RecommendItem(p24,pn24,,36.0), RecommendItem(p5,pn5,,55.0), RecommendItem(p13,pn13,,47.0), RecommendItem(p22,pn22,,38.0), RecommendItem(p52,pn52,,8.0), RecommendItem(p50,pn50,,10.0), RecommendItem(p26,pn26,,34.0), RecommendItem(p33,pn33,,27.0), RecommendItem(p45,pn45,,15.0), RecommendItem(p54,pn54,,6.0), RecommendItem(p29,pn29,,31.0), RecommendItem(p19,pn19,,41.0), RecommendItem(p32,pn32,,28.0), RecommendItem(p43,pn43,,17.0), RecommendItem(p42,pn42,,18.0), RecommendItem(p48,pn48,,12.0), RecommendItem(p30,pn30,,30.0), RecommendItem(p9,pn9,,51.0), RecommendItem(p51,pn51,,9.0), RecommendItem(p27,pn27,,33.0), RecommendItem(p17,pn17,,43.0), RecommendItem(p40,pn40,,20.0), RecommendItem(p53,pn53,,7.0), RecommendItem(p38,pn38,,22.0), RecommendItem(p56,pn56,,4.0), RecommendItem(p59,pn59,,1.0), RecommendItem(p57,pn57,,3.0), RecommendItem(p55,pn55,,5.0))

```
