package org.apache.spark.ml.tmp

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import java.util.concurrent.{Callable, Executors, ThreadPoolExecutor, Future}

object test {


  def main(args: Array[String]): Unit = {
    // 这一段是无法运行的，摘自 https://blog.csdn.net/u010454030/article/details/74353886

    val sparkConf = new SparkConf()
    sparkConf.setAppName("multi task submit ")
    sparkConf.setMaster("local[*]")
    //实例化spark context
    val sc = new SparkContext(sparkConf)

    //保存任务返回值
    var list = Array[Future[String]]()
    //并行任务读取的path
    var task_paths = Array[String]()
    task_paths :+= "C:/Users/linyigang/Desktop/大作业"
    task_paths :+= "C:/Users/linyigang/Desktop/大作业"
    task_paths :+= "C:/Users/linyigang/Desktop/大作业"
    println(task_paths)

    //线程数等于path的数量
    val nums_threads = task_paths.length
    //构建线程池
    val executors = Executors.newFixedThreadPool(nums_threads).asInstanceOf[ThreadPoolExecutor]
    for (i <- 0 until nums_threads) {
      val task = executors.submit(
        //        new Callable[String] {
        //          override def call(): String = {
        //            val count = sc.textFile(task_paths(i)).count() //获取统计文件数量
        //            return task_paths(i) + " 文件数量： " + count
        //          }
        //        }
        new MyGCForestTask(sc, task_paths(i))
      )

      list :+= task //添加集合里面
    }
    executors.shutdown()
    //遍历获取结果
    list.foreach(result => {
      println(result.get().toString)
    })
    //停止spark
    sc.stop()
    println(executors.getQueue)

    val queueSize = executors.getQueue().size();
    println("当前排队线程数:" + queueSize);
    val activeCount = executors.getActiveCount();
    println("当前活动线程数:" + activeCount);
    val completeTaskCount = executors.getCompletedTaskCount();
    println("执行完成线程数:" + completeTaskCount);
    val taskCount = executors.getTaskCount();
    println("总线程数:" + taskCount);
  }
}

class MyGCForestTask(sc: SparkContext, rspBlock: String) extends Callable[String] {
  override def call(): String = {
    val count = sc.textFile(rspBlock).count() //获取统计文件数量
    return rspBlock + " 文件数量： " + count
  }
}