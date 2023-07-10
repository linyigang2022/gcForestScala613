/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.examples

import org.apache.spark.ml.classification.{BLBGCForestClassifier, GCForestClassifier, RandomForestClassifier613 => RandomForestClassifier}
import org.apache.spark.ml.datasets._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.util.engine.Engine
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.util.SizeEstimator


object BLBGCForestSequence {
  def main(args: Array[String]): Unit = {

    import Utils._
    val stime = System.currentTimeMillis()
    val spark = SparkSession
      .builder()
      .appName(this.getClass.getSimpleName)
      .master("local[*]")
      .getOrCreate()

    val parallelism = Engine.getParallelism(spark.sparkContext)
    println(s"Total Cores is $parallelism")
    spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    spark.sparkContext.getConf.registerKryoClasses(Array(classOf[RandomForestClassifier]))
    println(trainParser.parse(args, TrainParams()).size)
    val param = trainParser.parse(args, TrainParams()).get

    spark.sparkContext.setLogLevel(param.debugLevel)
    spark.sparkContext.setCheckpointDir(param.checkpointDir)
    val output = param.model

    def getParallelism: Int = param.parallelism match {
      case p if p > 0 => param.parallelism
      case n if n < 0 => -1
      case _ => parallelism
    }

    val (train, test) = param.dataset match {
      case "uci_adult" => {
        val train = new UCI_adult().load_data(spark, param.trainFile, param.featuresFile, 1,
          getParallelism)
        val test = new UCI_adult().load_data(spark, param.testFile, param.featuresFile, 1,
          getParallelism)
        (train, test)
      }
      case "covertype" => {
        val data = new Covertype().load_data(spark, param.trainFile, param.featuresFile, 1,
          getParallelism)
        val Array(train, test) = data.randomSplit(Array(0.7, 0.3))
        (train, test)
      }
      case "watch_acc" => {
        val data = new WatchAcc().load_data(spark, param.trainFile, param.featuresFile, 1,
          getParallelism)
        val Array(train, test) = data.randomSplit(Array(0.7, 0.3))
        (train, test)
      }
      case "susy" => {
        val data = new SUSY().load_data(spark, param.trainFile, param.featuresFile, 1,
          getParallelism)
        val Array(train, test) = data.randomSplit(Array(0.7, 0.3))
        (train, test)
      }
      case "higgs" => {
        val data = new HIGGS().load_data(spark, param.trainFile, param.featuresFile, 1,
          getParallelism)
        val Array(train, test) = data.randomSplit(Array(0.7, 0.3))
        (train, test)
      }
    }
    train.printSchema()

    if (param.idebug) println(s"Estimate trainset %.1f M,".format(SizeEstimator.estimate(train) / 1048576.0) +
      s" testset: %.1f M".format(SizeEstimator.estimate(test) / 1048576.0))


    val blbGCForest = new BLBGCForestClassifier()
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
      .setSubRFNum(param.subRFNum)

    val model = blbGCForest.train(train, test)


    val totalTime = (System.currentTimeMillis() - stime) / 1000.0
    println(s"start to transform model")
    val out_test = model.transform(test)
    out_test.printSchema()
    out_test.withColumn("features", col("features").cast("string")).
      withColumn("rawPrediction", col("rawPrediction").cast("string")).
      withColumn("probability", col("probability").cast("string"))

    //      repartition(1).
    //    write.csv("out_test.csv")
    val concatUDF = udf { (features: Row) =>
      new DenseVector(features.getAs[Array[Double]]("features")++features.getAs[Array[Double]]("probability"))
    }
    out_test.select(concatUDF(col("features"),col("probability")).as("value")).show()
    //    out_test.write.format("csv").option("header", "true") save("myFile.csv")
    println(s"transformed result: $out_test, Sleep 20s")
    println(s"Total time for GCForest Application: $totalTime, Sleep 20s")

    Thread.sleep(20000)
    spark.stop()
  }
}

