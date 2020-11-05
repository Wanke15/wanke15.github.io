package com.bigdata.window

import java.sql.Timestamp
import java.util.Properties

import org.apache.flink.api.common.functions.AggregateFunction
import org.apache.flink.api.common.serialization.SimpleStringSchema
import org.apache.flink.api.common.state.{ListState, ListStateDescriptor}
import org.apache.flink.api.java.tuple.{Tuple, Tuple1, Tuple2}
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.TimeCharacteristic
import org.apache.flink.streaming.api.functions.KeyedProcessFunction
import org.apache.flink.streaming.api.scala.function.WindowFunction
import org.apache.flink.streaming.api.scala._
import org.apache.flink.streaming.api.windowing.assigners.{EventTimeSessionWindows, SlidingEventTimeWindows, TumblingEventTimeWindows}
import org.apache.flink.streaming.api.windowing.time.Time
import org.apache.flink.streaming.api.windowing.windows.TimeWindow
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer011
import org.apache.flink.util.Collector

import scala.collection.mutable.ListBuffer

object UserLikeItem {
  def main(args: Array[String]): Unit = {
    // 创建流处理的执行环境
    val env = StreamExecutionEnvironment.getExecutionEnvironment
    env.setParallelism(1)
    env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime)

    val properties = new Properties()
    properties.setProperty("bootstrap.servers", "localhost:9092")
    properties.setProperty("group.id", "flink-user-group")


    val inputStream = env.addSource( new FlinkKafkaConsumer011[String]("user_behavior", new SimpleStringSchema(), properties))

    val dataStream = inputStream.map(data => {
      val arr = data.split(",")
      UserItemBehavior(arr(0) + "_" + arr(1), arr(2).toLong, arr(3), arr(4).toLong)
    })
      .assignAscendingTimestamps(_.ts * 1000L)


    val aggStream = dataStream
      .keyBy("userItemId")
      .timeWindow(Time.days(1), Time.minutes(5))
      .aggregate(new UserBehaviorItemHeatAgg, new UserItemWindowResult)

    val resultStream = aggStream
      .keyBy("windowEnd")
      .process(new TopNHotItemsByUser(3))

    //    dataStream.print("dataStream")
    //    aggStream.print("aggStream")
    resultStream.print("resultStream")

    env.execute("user window test")
  }

}
case class UserItemBehavior(userItemId: String, catId: Long, behavType: String, ts: Long)

case class UserItemViewCount(userItemId: String, windowEnd: Long, likeScore: Long)

// COUNT统计的聚合函数实现，每出现一条记录就加一
class UserBehaviorItemHeatAgg extends AggregateFunction[UserItemBehavior, Long, Long] {
  override def createAccumulator(): Long = 0L
  override def add(userBehavior: UserItemBehavior, acc: Long): Long = {
    var score = 0
    if (userBehavior.behavType.equals("pv")){
      score = 1
    } else {
      if (userBehavior.behavType.equals("fav")){
        score = 2
      } else {
        if (userBehavior.behavType.equals("cart")){
          score = 3}
        else {
          if (userBehavior.behavType.equals("buy")){
            score = 4}
        }
      }
    }
    acc + score}
  override def getResult(acc: Long): Long = acc
  override def merge(acc1: Long, acc2: Long): Long = acc1 + acc2
}

//
class UserItemWindowResult extends WindowFunction[Long, UserItemViewCount, Tuple, TimeWindow] {
  override def apply(keys: Tuple, window: TimeWindow, aggregateResult: Iterable[Long],
                     collector: Collector[UserItemViewCount]) : Unit = {
    val userItemId: String = keys.asInstanceOf[Tuple1[String]].f0
    val likeScore = aggregateResult.iterator.next
    collector.collect(UserItemViewCount(userItemId, window.getEnd, likeScore))
  }
}

// 求某个窗口中前 N 名的热门点击商品，key 为窗口时间戳，输出为 TopN 的结果字符串
class TopNHotItemsByUser(topSize: Int) extends KeyedProcessFunction[Tuple, UserItemViewCount, String] {
  private var itemViewCountListState : ListState[UserItemViewCount] = _

  override def open(parameters: Configuration): Unit = {
    //    super.open(parameters)
    // 命名状态变量的名字和状态变量的类型
    //    val itemsStateDesc = new ListStateDescriptor[ItemViewCount]("itemState-state", classOf[ItemViewCount])
    // 从运行时上下文中获取状态并赋值
    itemViewCountListState = getRuntimeContext.getListState(new ListStateDescriptor[UserItemViewCount]("itemState-state", classOf[UserItemViewCount]))
  }

  override def processElement(value: UserItemViewCount, context: KeyedProcessFunction[Tuple, UserItemViewCount, String]#Context, collector: Collector[String]): Unit = {
    // 每条数据都保存到状态中
    //    println("***************** process *****************", value)
    //  val time: Long = System.currentTimeMillis()
    //      println("定时器第一次注册：" + time)
    itemViewCountListState.add(value)
    // 注册 windowEnd+1 的 EventTime Timer, 当触发时，说明收齐了属于windowEnd窗口的所有商品数据
    // 也就是当程序看到windowend + 1的水位线watermark时，触发onTimer回调函数
    context.timerService.registerEventTimeTimer(value.windowEnd + 1)
  }

  override def onTimer(timestamp: Long, ctx: KeyedProcessFunction[Tuple, UserItemViewCount, String]#OnTimerContext, out: Collector[String]): Unit = {
    // 字典保存不同user对不同item聚合统计的结果UserItemHeat
    var allItemsUserDict: Map[String, ListBuffer[UserItemHeat]] = Map()

    val iter = itemViewCountListState.get().iterator()
    while (iter.hasNext) {
      val curItem = iter.next()

      // userId_itemId 分割为 userId和itemId
      val arr = curItem.userItemId.split("_")
      
      // 在user已有的ListBuffer中add当前的数据
      var userCurItems = allItemsUserDict.getOrElse(arr(0), new ListBuffer[UserItemHeat])
      userCurItems += UserItemHeat(arr(0), arr(1), curItem.likeScore)
      
      allItemsUserDict += (arr(0) -> userCurItems)
    }

    itemViewCountListState.clear()

    val result: StringBuilder = new StringBuilder
    result.append("====================================\n")
    result.append("时间: ").append(new Timestamp(timestamp - 1)).append("\n")
    allItemsUserDict.keys.foreach{
      i => {
        // 当前用户的所有结果数据
        val allItems = allItemsUserDict(i)
        // 根据likeScore排序
        val sortedItems = allItems.sortBy(_.likeScore)(Ordering.Long.reverse).take(topSize)
        
        // 将排名信息格式化成 String, 便于打印
        for(i <- sortedItems.indices){
          val currentItem: UserItemHeat = sortedItems(i)
          // e.g.  商品ID=12224  No1： 浏览量=2413
          result
            .append("  用户ID=").append(currentItem.userId)
            .append("  No").append(i+1).append(":")
            .append("  商品ID=").append(currentItem.itemId)
            .append("  喜好度=").append(currentItem.likeScore).append("\n")
        }
      }
    }
    result.append("====================================\n\n")
    // 控制输出频率，模拟实时滚动结果
    Thread.sleep(5000)
    out.collect(result.toString)
  }
}

case class UserItemHeat(userId: String, itemId: String, likeScore: Long)
