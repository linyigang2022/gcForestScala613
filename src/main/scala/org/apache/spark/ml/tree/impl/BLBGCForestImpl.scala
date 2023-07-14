package org.apache.spark.ml.tree.impl


import java.text.SimpleDateFormat
import java.util.Date
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Helper.{UserDefinedFunctions => UDF}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.datasets.{Covertype, HIGGS, SUSY, UCI_adult, WatchAcc}
import org.apache.spark.ml.evaluation.{Accuracy, Metric, gcForestEvaluator}
import org.apache.spark.ml.examples.Utils.{TrainParams, trainParser}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector, VectorUDT}
import org.apache.spark.ml.tree.configuration.GCForestStrategy
import org.apache.spark.ml.tree.impl.GCForestImpl.getNowTime
import org.apache.spark.ml.util.engine.Engine
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DoubleType, LongType, StructField, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.SizeEstimator

import java.util.concurrent.{Callable, Executors, Future, ThreadPoolExecutor}


private[spark] object BLBGCForestImpl extends Logging {

  def run(
           input: Dataset[_],
           gcforestStategy: GCForestStrategy
         ): BLBGCForestClassificationModel = {
    train(input, strategy = gcforestStategy)
  }

  def runWithValidation(
                         input: Dataset[_],
                         validationInput: Dataset[_],
                         gCForestStrategy: GCForestStrategy
                       ): BLBGCForestClassificationModel = {
    trainWithValidation(input, validationInput, gCForestStrategy)
  }

  val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss,SSS")

  /**
   * Scan Sequence Data
   *
   * @param dataset    raw input label and features
   * @param windowSize the window size
   * @return
   */

  // create a random forest classifier by type
  def genRFClassifier(rfType: String,
                      strategy: GCForestStrategy,
                      isScan: Boolean,
                      num: Int): GCForestClassifier = {
    val rf = rfType match {
      case "rfc" => new GCForestClassifier().setRFNum(strategy.subRFNum).setCRFNum(0).setMaxIteration(1)
      case "crfc" => new GCForestClassifier().setRFNum(0).setCRFNum(strategy.subRFNum).setMaxIteration(1)
    }
    val gcForest = rf
      .setDataSize(strategy.dataSize)
      .setDataStyle(strategy.dataStyle)
      .setMultiScanWindow(strategy.multiScanWindow)
      .setCascadeForestTreeNum(strategy.cascadeForestTreeNum)
      .setScanForestTreeNum(strategy.scanForestTreeNum)
      .setMaxDepth(strategy.maxDepth)
      .setMaxBins(strategy.maxBins)
      .setMinInfoGain(strategy.minInfoGain)
      .setMaxMemoryInMB(strategy.maxMemoryInMB)
      .setCacheNodeId(strategy.cacheNodeId)
      .setScanForestMinInstancesPerNode(strategy.scanMinInsPerNode)
      .setCascadeForestMinInstancesPerNode(strategy.cascadeMinInsPerNode)
      .setFeatureSubsetStrategy(strategy.featureSubsetStrategy)
      .setCrf_featureSubsetStrategy(strategy.crf_featureSubsetStrategy)
      .setEarlyStoppingRounds(strategy.earlyStoppingRounds)
      .setIDebug(strategy.idebug)
      .setNumClasses(strategy.classNum)

    gcForest
  }

  /**
   * Concat multi-scan features
   *
   * @param dataset one of a window
   * @param sets    the others
   * @return input for Cascade Forest
   */
  def concatenate(
                   strategy: GCForestStrategy,
                   dataset: Dataset[_],
                   sets: Dataset[_]*
                 ): DataFrame = {
    val sparkSession = dataset.sparkSession
    var unionSet = dataset.toDF()
    sets.foreach(ds => unionSet = unionSet.union(ds.toDF()))

    class Record(val instance: Long, // instance id
                 val label: Double, // label
                 val features: Vector, // features
                 val scanId: Int, // the scan id for multi-scan
                 val forestId: Int, // forest id
                 val winId: Long) extends Serializable // window id

    val concatData = unionSet.select(
      strategy.instanceCol, strategy.labelCol,
      strategy.featuresCol, strategy.scanCol,
      strategy.forestIdCol, strategy.winCol).rdd.map {
      row =>
        val instance = row.getAs[Long](strategy.instanceCol)
        val label = row.getAs[Double](strategy.labelCol)
        val features = row.getAs[Vector](strategy.featuresCol)
        val scanId = row.getAs[Int](strategy.scanCol)
        val forestId = row.getAs[Int](strategy.forestIdCol)
        val winId = row.getAs[Long](strategy.winCol)

        new Record(instance, label, features, scanId, forestId, winId)
    }.groupBy(
      record => record.instance
    ).map { group =>
      val instance = group._1
      val records = group._2
      val label = records.head.label

      def recordCompare(left: Record, right: Record): Boolean = {
        var code = left.scanId.compareTo(right.scanId)
        if (code == 0) code = left.forestId.compareTo(right.forestId)
        if (code == 0) code = left.winId.compareTo(right.winId)
        code < 0
      }

      val features = new DenseVector(records.toSeq.sortWith(recordCompare)
        .flatMap(_.features.toArray).toArray)
      // features = [0, 0, ..., 0] (903 dim)
      Row.fromSeq(Array[Any](instance, label, features))
    }

    val schema: StructType = StructType(Seq[StructField]())
      .add(StructField(strategy.instanceCol, LongType))
      .add(StructField(strategy.labelCol, DoubleType))
      .add(StructField(strategy.featuresCol, new VectorUDT))
    sparkSession.createDataFrame(concatData, schema)
  }

  /**
   * concat inputs of Cascade Forest with prediction
   *
   * @param feature input features
   * @param predict prediction features
   * @return
   */
  def mergeFeatureAndPredict(
                              feature: Dataset[_],
                              predict: Dataset[_],
                              strategy: GCForestStrategy): DataFrame = {
    val vectorMerge = udf { (v1: Vector, v2: Vector) =>
      new DenseVector(v1.toArray.union(v2.toArray))
    }

    if (predict != null) {
      feature.join(
        // join (predict feature col to predictionCol)
        predict.withColumnRenamed(strategy.featuresCol, strategy.predictionCol),
        Seq(strategy.instanceCol) // join on instanceCol
        // add a featureCol with featureCol + predictionCol
      ).withColumn(strategy.featuresCol, vectorMerge(col(strategy.featuresCol),
        col(strategy.predictionCol))
      ).select(strategy.instanceCol, strategy.featuresCol, strategy.labelCol).toDF()
      // select 3 cols to DataFrame
    } else {
      feature.toDF()
    }
  }

  private def getNowTime = dateFormat.format(new Date())

  /**
   * Multi-Grained Scanning
   */
  def multi_grain_Scan(
                        dataset: Dataset[_],
                        strategy: GCForestStrategy): DataFrame = {

    require(dataset != null, "Null dataset need not to scan")
    // scalastyle:off println
    var scanFeature: DataFrame = null
    val rand = new Random()
    rand.setSeed(System.currentTimeMillis())

    println(s"[$getNowTime] Multi Grained Scanning begin!")

    if (strategy.dataStyle == "Seq") {
      scanFeature = dataset.toDF()
    }
    // scanFeature: (instanceId, label, features)
    println(s"[$getNowTime] Multi Grained Scanning finished!")
    // scalastyle:on println
    scanFeature
  }

  def train(
             input: Dataset[_],
             strategy: GCForestStrategy): BLBGCForestClassificationModel = {
    val numClasses: Int = strategy.classNum
    val erfModels = ArrayBuffer[Array[GCForestClassificationModel]]()
    val n_train = input.count()

    val scanFeature_train = multi_grain_Scan(input, strategy)

    scanFeature_train.cache()
    // scalastyle:off println
    println(s"[$getNowTime] Cascade Forest begin!")

    val sparkSession = scanFeature_train.sparkSession
    val sc = sparkSession.sparkContext
    val rng = new Random()
    rng.setSeed(System.currentTimeMillis())

    var lastPrediction: DataFrame = null
    val acc_list = ArrayBuffer[Double]()

    // Init classifiers
    val maxIteration = strategy.maxIteration
    require(maxIteration > 0, "Non-positive maxIteration")
    var layer_id = 1
    var reachMaxLayer = false

    while (!reachMaxLayer) {

      println(s"[$getNowTime] Training Cascade Forest Layer $layer_id")

      val gcForests = (
        Range(0, 4).map(it =>
          genRFClassifier("rfc", strategy, isScan = false, num = rng.nextInt + it))
          ++
          Range(4, 8).map(it =>
            genRFClassifier("crfc", strategy, isScan = false, num = rng.nextInt + it))
        ).toArray[GCForestClassifier]
      assert(gcForests.length == 8, "random Forests inValid!")
      // scanFeatures_*: (instanceId, label, features)
      val training = mergeFeatureAndPredict(scanFeature_train, lastPrediction, strategy)
        .persist(StorageLevel.MEMORY_ONLY_SER)
      val bcastTraining = sc.broadcast(training)
      val features_dim = training.first().mkString.split(",").length

      println(s"[$getNowTime] Training Set = ($n_train, $features_dim)")

      var ensemblePredict: DataFrame = null // closure need

      var layer_train_metric: Accuracy = new Accuracy(0, 0) // closure need

      println(s"[$getNowTime] Forests fitting and transforming ......")

      erfModels += gcForests.zipWithIndex.map { case (gcForest, it) =>
        gcForest.train(training)
      }

      println(s"[$getNowTime] [Layer Summary] layer [$layer_id] - " +
        s"train.classifier.average = ${layer_train_metric.div(8d)}")
      println(s"[$getNowTime] Forests fitting and transforming finished!")

      val schema = new StructType()
        .add(StructField(strategy.instanceCol, LongType))
        .add(StructField(strategy.featuresCol, new VectorUDT))

      println(s"[$getNowTime] Getting prediction RDD ......")

      acc_list += layer_train_metric.getAccuracy


      val predictRDDs = {
        val grouped = ensemblePredict.rdd.groupBy(_.getAs[Long](strategy.instanceCol))
        grouped.map { group =>
          val instanceId = group._1
          val rows = group._2
          val features = new DenseVector(rows.toArray
            .sortWith(_.getAs[Int](strategy.forestIdCol) < _.getAs[Int](strategy.forestIdCol))
            .flatMap(_.getAs[Vector](strategy.featuresCol).toArray))
          Row.fromSeq(Array[Any](instanceId, features))
        }
      }
      // predictRDDs.foreach(r => r.persist(StorageLevel.MEMORY_ONLY_SER))
      println(s"[$getNowTime] Get prediction RDD finished! Layer $layer_id training finished!")

      val opt_layer_id_train = acc_list.zipWithIndex.maxBy(_._1)._2


      if (opt_layer_id_train + 1 == layer_id) {
        println(s"[$getNowTime] [Result] [Optimal Layer] max_layer_num = $layer_id " +
          s"accuracy_train = ${acc_list(opt_layer_id_train) * 100}%")
      }
      lastPrediction = sparkSession.createDataFrame(predictRDDs, schema).cache()
      val outOfRounds = layer_id - opt_layer_id_train >= strategy.earlyStoppingRounds
      if (outOfRounds) {
        println(s"[$getNowTime] " +
          s"[Result][Optimal Level Detected] opt_layer_id = " +
          s"$opt_layer_id_train, " +
          s"accuracy_train=${acc_list(opt_layer_id_train)}")
      }
      reachMaxLayer = (layer_id == maxIteration) || outOfRounds
      if (reachMaxLayer) {
        println(s"[$getNowTime] " +
          s"[Result][Reach Max Layer] max_layer_num=$layer_id, " +
          s"accuracy_train=$layer_train_metric")
      }
      layer_id += 1
    }

    scanFeature_train.unpersist()

    println(s"[$getNowTime] Cascade Forest Training Finished!")
    // scalastyle:on println
    new BLBGCForestClassificationModel(erfModels.toArray, numClasses)
  }

  /**
   * Train a Cascade Forest
   */
  private def trainWithValidation(
                                   input: Dataset[_],
                                   validationInput: Dataset[_],
                                   strategy: GCForestStrategy): BLBGCForestClassificationModel = {
    val timer = new TimeTracker()
    timer.start("total")
    // scalastyle:off println
    if (strategy.idebug) println(s"[$getNowTime] timer.start(total)")
    val numClasses: Int = strategy.classNum
    // TODO: better representation to model
    val erfModels = ArrayBuffer[Array[GCForestClassificationModel]]() // layer - (forest * fold)
    val n_train = input.count()
    val n_test = validationInput.count()

    timer.start("multi_grain_Scan for Train and Test")
    val scanFeature_train = multi_grain_Scan(input, strategy)
    val scanFeature_test = multi_grain_Scan(validationInput, strategy)
    println(s"input:${input}")
    println(s"scanFeature_train:$scanFeature_train")
    timer.stop("multi_grain_Scan for Train and Test")

    timer.start("cache scanFeature of Train and Test")
    scanFeature_train.cache()
    scanFeature_test.cache()
    timer.stop("cache scanFeature of Train and Test")

    println(s"[$getNowTime] Cascade Forest begin!")

    timer.start("init")
    if (strategy.idebug) println(s"[$getNowTime] timer.start(init)")
    val sparkSession = scanFeature_train.sparkSession
    val sc = sparkSession.sparkContext
    val rng = new Random()
    rng.setSeed(System.currentTimeMillis())

    var lastPrediction: DataFrame = null
    var lastPrediction_test: DataFrame = null
    val acc_list = Array(ArrayBuffer[Double](), ArrayBuffer[Double]())
    var ensemblePredict: DataFrame = null // closure need
    var ensemblePredict_test: DataFrame = null // closure need

    var layer_train_metric: Accuracy = new Accuracy(0, 0) // closure need
    var layer_test_metric: Accuracy = new Accuracy(0, 0) // closure need

    // Init classifiers
    val maxIteration = strategy.maxIteration
    require(maxIteration > 0, "Non-positive maxIteration")
    var layer_id = 1
    var reachMaxLayer = false

    timer.stop("init")
    if (strategy.idebug) println(s"[$getNowTime] timer.stop(init)")
    println("创建线程池")
    val executors = Executors.newFixedThreadPool(strategy.rfNum + strategy.crfNum).asInstanceOf[ThreadPoolExecutor]
    while (!reachMaxLayer) {
      if (strategy.idebug)
        println(s"sc.defaultParallelism = ${sc.defaultParallelism}")

      val stime = System.currentTimeMillis()

      println(s"[$getNowTime] Training Cascade Forest Layer $layer_id")

      val gcForests = (
        Range(0, strategy.rfNum).map(it =>
          genRFClassifier("rfc", strategy, isScan = false, num = rng.nextInt + it))
          ++
          Range(strategy.rfNum, strategy.rfNum + strategy.crfNum).map(it =>
            genRFClassifier("crfc", strategy, isScan = false, num = rng.nextInt + it))
        ).toArray[GCForestClassifier]
      assert(gcForests.length == strategy.rfNum + strategy.crfNum, "blb random Forests inValid!")
      // scanFeatures_*: (instanceId, label, features)
      timer.start("merge to produce training, testing and persist")
      val training = mergeFeatureAndPredict(scanFeature_train, lastPrediction, strategy)
        .coalesce(sc.defaultParallelism)

      val testing = mergeFeatureAndPredict(scanFeature_test, lastPrediction_test, strategy)
        .coalesce(sc.defaultParallelism)

      timer.stop("merge to produce training, testing and persist")
      if (strategy.idebug) println(s"[$getNowTime]" +
        s" timer.stop(merge to produce training, testing and persist)")

      val features_dim = training.first().mkString.split(",").length // action, get training truly

      println(s"[$getNowTime] Training Set = ($n_train, $features_dim), " +
        s"Testing Set = ($n_test, $features_dim)")


      ensemblePredict = null
      ensemblePredict_test = null

      layer_train_metric.reset()
      layer_test_metric.reset()
      // lyg
      //      val weight = 1.0 / (strategy.rfNum + strategy.crfNum)
      //      val weightArray = Array.fill[Double](strategy.subRFNum)(weight)
//            val sampleTraining = training.randomSplit(weightArray)
      //      val sampleTesting = testing.randomSplit(weightArray)
      val n = training.count().toDouble
//      val lambda = 0.6
      val lambda = strategy.lambda
      println(s"当前设定的lambda值为：${lambda}")
      val frac = math.pow(n, lambda) / n
      val sampleTraining = Array.ofDim[Dataset[Row]](strategy.rfNum + strategy.crfNum)
      val sampleTesting = Array.ofDim[Dataset[Row]](strategy.rfNum + strategy.crfNum)
      for (i <- 0 until strategy.rfNum + strategy.crfNum){
        sampleTraining(i) = training.sample(false, frac)
        sampleTesting(i) = testing.sample(false, frac)
      }
      println(s"gcforest子森林采用有放回的抽样，原本大小为${training.count()},抽样后的大小为${sampleTraining(0).count()}")
      println(s"子森林数量：${strategy.rfNum + strategy.crfNum}")
      println(s"strategy.crfNum：${strategy.subRFNum}")
      sampleTraining.foreach { training =>
        training.cache()
      }
      sampleTesting.foreach { testing =>
        testing.cache()
      }

      println(s"[$getNowTime] Forests fitting and transforming ......")
      timer.start("randomForests training")
      if (strategy.idebug) println(s"[$getNowTime] timer.start(randomForests training)")

      //-----------------------------------------------------------
      //并行地对training和testing进行transform，权衡通信成本和速度
      var list = Array[Future[(DataFrame, DataFrame, GCForestClassificationModel)]]()
      val executors = Executors.newFixedThreadPool(strategy.rfNum + strategy.crfNum).asInstanceOf[ThreadPoolExecutor]
      gcForests.zipWithIndex.foreach { case (gcforest, it) =>
        println(s"[$getNowTime] blb layer [${layer_id}] gcForests subforest [${it}] fitting and transforming ......")
        if (strategy.idebug) println(s"[$getNowTime] timer.start(cvClassVectorGeneration)")

        //        val model = gcforest.train(sampleTraining(it), sampleTesting(it))
        val task = executors.submit(
          new GCForestTask(sparkSession, gcforest, sampleTraining(it), sampleTesting(it), training, testing)
        )
        list :+= task //添加集合里面
      }
      executors.shutdown()
      val totalTime = (System.currentTimeMillis() - stime) / 1000.0
      println(s"任务已提交，等待执行")
      //遍历获取结果
      val transformed = list.map(result => {
        val transformed = result.get()
        println(transformed.toString)
        transformed
      })

      println(s"Total time for GCForest Application: $totalTime, 开始最后转换")
      erfModels += transformed.map {
        _._3
      }.toArray
      transformed.zipWithIndex.map { case (t, it) =>
        val schema = new StructType()
          .add(StructField(strategy.instanceCol, LongType))
          .add(StructField(strategy.featuresCol, new VectorUDT))

        val predict = t._1.drop(strategy.featuresCol)
          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
          .withColumn(strategy.forestIdCol, lit(it))
          .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol)

        val predict_test = t._2.drop(strategy.featuresCol)
          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
          .withColumn(strategy.forestIdCol, lit(it))
          .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol)
        ensemblePredict =
          if (ensemblePredict == null) predict else ensemblePredict.union(predict)
        ensemblePredict_test =
          if (ensemblePredict_test == null) predict_test else ensemblePredict_test.union(predict_test)

        val train_result = t._1.drop(strategy.featuresCol)
          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        val test_result = t._2.drop(strategy.featuresCol)
          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
        val train_acc = gcForestEvaluator.evaluatePartition(train_result)
        val test_acc = gcForestEvaluator.evaluatePartition(test_result)
        layer_train_metric = layer_train_metric + train_acc
        layer_test_metric = layer_test_metric + test_acc
      }
      //-----------------------------------------------------------
      //      //串行地对training和testing进行transform，权衡通信成本和速度
      //      var list = Array[Future[GCForestClassificationModel]]()
      //
      //      gcForests.zipWithIndex.foreach { case (gcforest, it) =>
      //        println(s"[$getNowTime] blb layer [${layer_id}] gcForests subforest [${it}] fitting and transforming ......")
      //        if (strategy.idebug) println(s"[$getNowTime] timer.start(cvClassVectorGeneration)")
      //
      ////        val model = gcforest.train(sampleTraining(it), sampleTesting(it))
      //        val task = executors.submit(
      //          new GCForestTask2(sparkSession, gcforest, sampleTraining(it), sampleTesting(it))
      //        )
      //        list :+= task //添加集合里面
      //      }
      //      val totalTime = (System.currentTimeMillis() - stime) / 1000.0
      //      println(s"任务已提交，等待执行")
      //      //遍历获取结果
      //      val erfModel = list.map(result => {
      //        val model = result.get()
      //        println(model.toString)
      //        model
      //      })
      //
      //      println(s"Total time for GCForest Application: $totalTime, 开始最后转换")
      //      erfModels += erfModel
      //      erfModel.zipWithIndex.map { case (model, it) =>
      //        val transformedTrain = model.transform(training).drop(strategy.featuresCol)
      //          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
      //        val transformedTest = model.transform(testing).drop(strategy.featuresCol)
      //          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
      //        val predict = transformedTrain
      //          .withColumn(strategy.forestIdCol, lit(it))
      //          .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol)
      //
      //        val predict_test = transformedTest
      //          .withColumn(strategy.forestIdCol, lit(it))
      //          .select(strategy.instanceCol, strategy.featuresCol, strategy.forestIdCol)
      //        ensemblePredict =
      //          if (ensemblePredict == null) predict else ensemblePredict.union(predict)
      //        ensemblePredict_test =
      //          if (ensemblePredict_test == null) predict_test else ensemblePredict_test.union(predict_test)
      //
      //        val train_result = transformedTrain
      //          .drop(strategy.featuresCol)
      //          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
      //        val test_result = transformedTest
      //          .drop(strategy.featuresCol)
      //          .withColumnRenamed(strategy.probabilityCol, strategy.featuresCol)
      //        val train_acc = gcForestEvaluator.evaluatePartition(train_result)
      //        val test_acc = gcForestEvaluator.evaluatePartition(test_result)
      //        layer_train_metric = layer_train_metric + train_acc
      //        layer_test_metric = layer_test_metric + test_acc
      //      }
      //-----------------------------------------------------------
      timer.stop("randomForests training")

      sampleTraining.foreach { training =>
        training.unpersist(blocking = true)
      }
      sampleTesting.foreach { testing =>
        testing.unpersist(blocking = true)
      }

      println(s"[$getNowTime] Forests fitting and transforming finished!")

      acc_list(0) += layer_train_metric.getAccuracy
      acc_list(1) += layer_test_metric.getAccuracy

      println(s"[$getNowTime] Getting prediction RDD ......")
      timer.start("flatten prediction")
      if (strategy.idebug) println(s"[$getNowTime] timer.start(flatten prediction)")
      val predictRDDs =
        Array(ensemblePredict, ensemblePredict_test).map { predict =>
          val grouped = predict.rdd.groupBy(_.getAs[Long](strategy.instanceCol))
          val predictRDD = grouped.map { group =>
            val instanceId = group._1
            val rows = group._2
            val features = new DenseVector(rows.toArray
              .sortWith(_.getAs[Int](strategy.forestIdCol) < _.getAs[Int](strategy.forestIdCol))
              .flatMap(_.getAs[Vector](strategy.featuresCol).toArray))
            Row.fromSeq(Array[Any](instanceId, features))
          }
          //          val schema = new StructType()
          //            .add(StructField(strategy.instanceCol, LongType))
          //            .add(StructField(strategy.featuresCol, new VectorUDT))
          //          sparkSession.createDataFrame(predictRDD, schema).printSchema()
          predictRDD
        }
      val predictRDDDim = predictRDDs(0).first().mkString.split(",").length

      println(s"[$getNowTime] blb gcforestImpl layer train finish, predict rdd feature dim = ($predictRDDDim)")
      timer.stop("flatten prediction")
      if (strategy.idebug) println(s"[$getNowTime] timer.stop(flatten prediction)")
      println(s"[$getNowTime] Get prediction RDD finished! Layer $layer_id training finished!")

      val opt_layer_id_train = acc_list(0).zipWithIndex.maxBy(_._1)._2
      val opt_layer_id_test = acc_list(1).zipWithIndex.maxBy(_._1)._2

      if (strategy.earlyStopByTest) {
        if (opt_layer_id_test + 1 == layer_id) {
          println(s"[$getNowTime] blb layer [${layer_id}] gcForests summary" +
            s"[Result] [Optimal Layer] opt_layer_num = $layer_id " +
            "accuracy_train=%.3f%%, ".format(acc_list(0)(opt_layer_id_train) * 100) +
            "accuracy_test=%.3f%%".format(acc_list(1)(opt_layer_id_test) * 100))
//            s"accuracy_train=${acc_list(0)}, accuracy_test=${acc_list(1)}")
        }
      }
      else {
        if (opt_layer_id_train + 1 == layer_id) {
          println(s"[$getNowTime] blb layer [${layer_id}] gcForests summary" +
            s"[Result] [Optimal Layer] opt_layer_num = $layer_id " +
            "accuracy_train=%.3f%%, ".format(acc_list(0)(opt_layer_id_train) * 100) +
            "accuracy_test=%.3f%%".format(acc_list(1)(opt_layer_id_test) * 100))
        }
      }
      if (strategy.idebug) {
        println(s"[$getNowTime] Not Persist but calculate lastPrediction and lastPrediction_test")
      }
      val schema = new StructType()
        .add(StructField(strategy.instanceCol, LongType))
        .add(StructField(strategy.featuresCol, new VectorUDT))
      lastPrediction = sparkSession.createDataFrame(predictRDDs(0), schema)
        .coalesce(sc.defaultParallelism)

      lastPrediction_test = sparkSession.createDataFrame(predictRDDs(1), schema)
        .coalesce(sc.defaultParallelism)

      val vectorMerge = udf { (v1: Vector) =>
        val avgPredict = Array.fill[Double](numClasses)(0d)
        val lastPredict = v1.toArray
        lastPredict.indices.foreach { i =>
          val classType = i % numClasses
          avgPredict(classType) = avgPredict(classType) + lastPredict(i)
        }
        new DenseVector(avgPredict)
      }
      lastPrediction = lastPrediction.withColumn(strategy.featuresCol, vectorMerge(col(strategy.featuresCol)))
      lastPrediction_test = lastPrediction_test.withColumn(strategy.featuresCol, vectorMerge(col(strategy.featuresCol)))
      println(s"特征堆叠后的lastPrediction dim：${lastPrediction.first().mkString.split(",").length}")

      val outOfRounds = (strategy.earlyStopByTest &&
        layer_id - opt_layer_id_test >= strategy.earlyStoppingRounds) ||
        (!strategy.earlyStopByTest && layer_id - opt_layer_id_train >= strategy.earlyStoppingRounds)
      if (outOfRounds) {
        println(s"[$getNowTime] " +
          s"[Result][Optimal Level Detected] opt_layer_id = " +
          s"${if (strategy.earlyStopByTest) opt_layer_id_test else opt_layer_id_train}, " +
          "accuracy_train=%.3f %%, ".format(acc_list(0)(opt_layer_id_train) * 100) +
          "accuracy_test=%.3f %%".format(acc_list(1)(opt_layer_id_test) * 100))
      }
      reachMaxLayer = (layer_id == maxIteration) || outOfRounds
      if (reachMaxLayer) {
        println(s"[$getNowTime] " +
          s"[Result][Reach Max Layer] max_layer_num=$layer_id, " +
          s"accuracy_train=$layer_train_metric, accuracy_test=$layer_test_metric")
      }
      println(s"[$getNowTime] Layer $layer_id" +
        s" time cost: ${(System.currentTimeMillis() - stime) / 1000.0} s")
      layer_id += 1
      if (strategy.idebug) {
        println(s"Layer ${layer_id - 1} Time Summary")
        println(s"$timer")
      }
    }
    println("关闭线程池")
    executors.shutdown()
    scanFeature_train.unpersist()
    scanFeature_test.unpersist()

    println(s"[$getNowTime] Cascade Forest Training Finished!")
    timer.stop("total")
    if (strategy.idebug) println(s"[$getNowTime] timer.stop(total)")

    println(s"[$getNowTime] Internal timing for GCForestImpl:")
    println(s"$timer")
    // scalastyle:on println
    new BLBGCForestClassificationModel(erfModels.toArray, numClasses)
  }
}

class GCForestTask(spark: SparkSession, gcForest: GCForestClassifier, sampleTrain: Dataset[_], sampleTest: Dataset[_], train: Dataset[_], test: Dataset[_])
  extends Callable[(DataFrame, DataFrame, GCForestClassificationModel)] {
  override def call(): (DataFrame, DataFrame, GCForestClassificationModel) = {
    val sc = spark.sparkContext
    val parallelism = Engine.getParallelism(spark.sparkContext)
    println(s"Total Cores is $parallelism")
    spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    spark.sparkContext.getConf.registerKryoClasses(Array(classOf[RandomForestClassifier]))

    val model = gcForest.train(sampleTrain, sampleTest)
    val transformedTrain = model.transform(train)
    val transformedTest = model.transform(test)
    println(s"multithread training gcforest Finished!")
    (transformedTrain, transformedTest, model)
  }
}

class GCForestTask2(spark: SparkSession, gcForest: GCForestClassifier, sampleTrain: Dataset[_], sampleTest: Dataset[_])
  extends Callable[GCForestClassificationModel] {
  override def call(): GCForestClassificationModel = {
    val sc = spark.sparkContext
    val parallelism = Engine.getParallelism(spark.sparkContext)
    println(s"Total Cores is $parallelism")
    spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    spark.sparkContext.getConf.registerKryoClasses(Array(classOf[RandomForestClassifier]))

    val model = gcForest.train(sampleTrain, sampleTest)
    println(s"multithread training gcforest Finished!")
    model
  }
}