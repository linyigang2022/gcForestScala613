package org.apache.spark.ml.tmp

import org.apache.spark.ml.classification.{GCForestClassificationModel, GCForestClassifier, RandomForestClassifier}
import org.apache.spark.ml.datasets.UCI_adult
import org.apache.spark.ml.examples.UCI_adult.Utils
import org.apache.spark.ml.examples.UCI_adult.Utils.{TrainParams, trainParser}
import org.apache.spark.ml.util.engine.Engine
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.SizeEstimator

import java.util.concurrent.{Callable, Executors, Future, ThreadPoolExecutor}

object ParallelGCForest {


  def main(args: Array[String]): Unit = {


//    val sparkConf = new SparkConf()
//    sparkConf.setAppName("multi task submit ")
//    sparkConf.setMaster("local[*]")
    //实例化spark context


    //保存任务返回值
    var list = Array[Future[Option[GCForestClassificationModel]]]()
    //并行任务读取的path
    var task_paths = Array[String]()
    task_paths :+= "1"
    task_paths :+= "2"
    task_paths :+= "3"
    task_paths :+= "4"

    import Utils._
    val stime = System.currentTimeMillis()
    val spark = SparkSession
      .builder()
      .appName(this.getClass.getSimpleName)
//      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()

    val parallelism = Engine.getParallelism(spark.sparkContext)
    println(s"Total Cores is $parallelism")


    //线程数等于path的数量
    val nums_threads = 4
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
        new GCForestTask(args, spark, task_paths(i))
      )
      list :+= task //添加集合里面
    }
    executors.shutdown()
    val totalTime = (System.currentTimeMillis() - stime) / 1000.0
    println(s"Total time for GCForest Application: $totalTime, Sleep 20s")
    //遍历获取结果
    list.foreach(result => {
      println(result.get().toString)
    })
    //停止spark
    spark.sparkContext.stop()
//    println(executors.getQueue)

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

class GCForestTask(args: Array[String], spark: SparkSession, rspBlock: String) extends Callable[Option[GCForestClassificationModel]] {
  override def call(): Option[GCForestClassificationModel] = {
    //    val count = sc.textFile(rspBlock).count() //获取统计文件数量
    //    return rspBlock + " 文件数量： " + count
    val sc = spark.sparkContext
    val parallelism = Engine.getParallelism(spark.sparkContext)
    println(s"Total Cores is $parallelism")
    //    spark.conf.set("spark.default.parallelism", parallelism)
    //    spark.conf.set("spark.locality.wait.node", 0)
    spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    //    spark.sparkContext.getConf.registerKryoClasses(Array(classOf[RandomForestCARTClassifier]))
    spark.sparkContext.getConf.registerKryoClasses(Array(classOf[RandomForestClassifier]))

    val gcForest = trainParser.parse(args, TrainParams()).map(param => {

      spark.sparkContext.setLogLevel(param.debugLevel)
      spark.sparkContext.setCheckpointDir(param.checkpointDir)
      val output = param.model

      def getParallelism: Int = param.parallelism match {
        case p if p > 0 => param.parallelism
        case n if n < 0 => -1
        case _ => parallelism
      }

      val train = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1,
        getParallelism)
      val test = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1,
        getParallelism)
      if (param.idebug) println(s"Estimate trainset %.1f M,".format(SizeEstimator.estimate(train) / 1048576.0) +
        s" testset: %.1f M".format(SizeEstimator.estimate(test) / 1048576.0))
      val gcForest = new GCForestClassifier()
        .setModelPath(param.model)
        .setDataSize(param.dataSize)
        .setDataStyle(param.dataStyle)
        .setMultiScanWindow(param.multiScanWindow)
        .setRFNum(param.rfNum)
        .setCRFNum(param.crfNum)
        .setCascadeForestTreeNum(param.cascadeForestTreeNum)
        .setScanForestTreeNum(param.scanForestTreeNum)
        .setMaxIteration(param.maxIteration)
        .setMaxDepth(param.maxDepth)
        .setMaxBins(param.maxBins)
        .setMinInfoGain(param.minInfoGain)
        .setMaxMemoryInMB(param.maxMemoryInMB)
        .setCacheNodeId(param.cacheNodeId)
        .setScanForestMinInstancesPerNode(param.scanMinInsPerNode)
        .setCascadeForestMinInstancesPerNode(param.cascadeMinInsPerNode)
        .setFeatureSubsetStrategy(param.featureSubsetStrategy)
        .setCrf_featureSubsetStrategy(param.crf_featureSubsetStrategy)
        .setEarlyStoppingRounds(param.earlyStoppingRounds)
        .setIDebug(param.idebug)

      val model = gcForest.train(train, test)
      model
    })
    println(s"training gcforest ${rspBlock} Finished!")
    gcForest
  }
}