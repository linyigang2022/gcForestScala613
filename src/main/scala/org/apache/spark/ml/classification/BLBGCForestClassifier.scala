package org.apache.spark.ml.classification

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.datasets._
import org.apache.spark.ml.examples.Utils.{TrainParams, trainParser}
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.GCForestParams
import org.apache.spark.ml.tree.configuration.GCForestStrategy
import org.apache.spark.ml.tree.impl.BLBGCForestImpl
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.engine.Engine
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.util.SizeEstimator

import java.util.concurrent.Callable

class BLBGCForestClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, BLBGCForestClassifier, BLBGCForestClassificationModel]
    with DefaultParamsWritable with GCForestParams {

  def this() = this(Identifiable.randomUID("gcf"))

  override def setNumClasses(value: Int): this.type = set(classNum, value)

  override def setModelPath(value: String): BLBGCForestClassifier.this.type = set(modelPath, value)

  override def setDataSize(value: Array[Int]): this.type = set(dataSize, value)

  override def setDataStyle(value: String): this.type = set(dataStyle, value)

  override def setMultiScanWindow(value: Array[Int]): this.type = set(multiScanWindow, value)

  override def setScanForestTreeNum(value: Int): this.type = set(scanForestTreeNum, value)

  override def setCascadeForestTreeNum(value: Int): this.type = set(cascadeForestTreeNum, value)

  override def setMaxIteration(value: Int): this.type = set(MaxIteration, value)

  override def setEarlyStoppingRounds(value: Int): this.type = set(earlyStoppingRounds, value)

  override def setIDebug(value: Boolean): BLBGCForestClassifier.this.type = set(idebug, value)

  override def setSubRFNum(value: Int): BLBGCForestClassifier.this.type = set(subRFNum, value)

  override def setLambda(value: Double): BLBGCForestClassifier.this.type = set(lambda, value)

  override def setMaxDepth(value: Int): this.type = set(MaxDepth, value)

  override def setMaxBins(value: Int): BLBGCForestClassifier.this.type = set(MaxBins, value)

  override def setMinInfoGain(value: Double): BLBGCForestClassifier.this.type = set(minInfoGain, value)

  override def setScanForestMinInstancesPerNode(value: Int):
  BLBGCForestClassifier.this.type = set(scanMinInsPerNode, value)

  override def setCascadeForestMinInstancesPerNode(value: Int):
  BLBGCForestClassifier.this.type = set(cascadeMinInsPerNode, value)

  override def setFeatureSubsetStrategy(value: String):
  BLBGCForestClassifier.this.type = set(featureSubsetStrategy, value)

  override def setCacheNodeId(value: Boolean):
  BLBGCForestClassifier.this.type = set(cacheNodeId, value)

  override def setMaxMemoryInMB(value: Int):
  BLBGCForestClassifier.this.type = set(maxMemoryInMB, value)

  override def setRFNum(value: Int): BLBGCForestClassifier.this.type = set(rfNum, value)

  override def setCRFNum(value: Int): BLBGCForestClassifier.this.type = set(crfNum, value)

  def getGCForestStrategy: GCForestStrategy = {
    GCForestStrategy($(classNum), $(modelPath), $(multiScanWindow),
      $(dataSize), $(rfNum), $(crfNum),
      $(scanForestTreeNum), $(cascadeForestTreeNum), $(scanMinInsPerNode),
      $(cascadeMinInsPerNode), $(featureSubsetStrategy), $(crf_featureSubsetStrategy), $(MaxBins),
      $(MaxDepth), $(minInfoGain), $(MaxIteration), $(maxMemoryInMB),
      $(numFolds), $(earlyStoppingRounds),
      $(earlyStopByTest), $(dataStyle), $(seed), $(cacheNodeId),
      $(windowCol), $(scanCol), $(forestIdCol), $(idebug),
      $(subRFNum), $(lambda))
  }

  def getDefaultStrategy: GCForestStrategy = {
    GCForestStrategy(2, $(modelPath), Array(), Array(113), idebug = false)
  }

  def train(trainset: Dataset[_], testset: Dataset[_]): BLBGCForestClassificationModel = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train().
    transformSchema(trainset.schema, logging = true)
    transformSchema(testset.schema, logging = true)

    // Cast LabelCol to DoubleType and keep the metadata.
    val labelMeta = trainset.schema($(labelCol)).metadata

    val casted_train =
      trainset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)

    val labelMeta_test = testset.schema($(labelCol)).metadata
    val casted_test =
      testset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta_test)

//    val gcForests = ArrayBuffer[GCForestClassificationModel]()
//    gcForests += copyValues(GCForestImpl.runWithValidation(casted_train, casted_test, getGCForestStrategy))
//    gcForests += copyValues(GCForestImpl.runWithValidation(casted_train, casted_test, getGCForestStrategy))
//    new BLBGCForestClassificationModel(gcForests.toArray)
    copyValues(BLBGCForestImpl.runWithValidation(casted_train, casted_test, getGCForestStrategy))
  }

  override def train(dataset: Dataset[_]): BLBGCForestClassificationModel = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train().
    transformSchema(dataset.schema, logging = true)

    // Cast LabelCol to DoubleType and keep the metadata.
    val labelMeta = dataset.schema($(labelCol)).metadata
    val casted_train =
      dataset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)
    BLBGCForestImpl.run(casted_train, getGCForestStrategy)
  }

  override def copy(extra: ParamMap): BLBGCForestClassifier = defaultCopy(extra)
}


private[ml] class BLBGCForestClassificationModel(
                                                  override val uid: String,
                                                  private val gcForests: Array[Array[GCForestClassificationModel]],
                                                  override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, BLBGCForestClassificationModel]
    with GCForestParams with MLWritable with Serializable {

  def this(
            cascadeForest: Array[Array[GCForestClassificationModel]],
            numClasses: Int) =
    this(Identifiable.randomUID("gcfc"), cascadeForest, numClasses)

  val numGCForests: Int = gcForests.length


  //  override def predictRaw(features: Vector): Vector = features
  override def predictRaw(features: Vector): Vector = {
    //  todo lyg
    var scanFeatures: Vector = null
    if ($(dataStyle) == "Seq") {
      scanFeatures = features
    }
    val avgPredict = Array.fill[Double](numClasses)(0d)
    var lastPredict = Array[Double]()

    gcForests.foreach { models =>
      lastPredict = models.flatMap(
        m => m.predictProbability(new DenseVector(features.toArray.union(lastPredict))).toArray
      )
    }

    lastPredict.indices.foreach { i =>
      val classType = i % numClasses
      avgPredict(classType) = avgPredict(classType) + lastPredict(i)
    }

    new DenseVector(avgPredict)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case _: SparseVector =>
        throw new RuntimeException("Unexpected error in BLBGCForestClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }



  override def write: MLWriter =
    new BLBGCForestClassificationModel.BLBGCForestClassificationModelWriter(this)


  override def copy(extra: ParamMap): BLBGCForestClassificationModel = {
    copyValues(new BLBGCForestClassificationModel(uid, gcForests, numClasses), extra)
  }
}


object BLBGCForestClassificationModel extends MLReadable[BLBGCForestClassificationModel] {
  override def read: MLReader[BLBGCForestClassificationModel] = new BLBGCForestClassificationModelReader

  override def load(path: String): BLBGCForestClassificationModel = super.load(path)

  private[BLBGCForestClassificationModel]
  class BLBGCForestClassificationModelWriter(instance: BLBGCForestClassificationModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
//      todo 实现模型的保存，如果需要再参照着写
    }
  }

  private class BLBGCForestClassificationModelReader
    extends MLReader[BLBGCForestClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BLBGCForestClassificationModel].getName

    override def load(path: String): BLBGCForestClassificationModel = {
      //      todo lyg 实现模型的加载
      implicit val format = DefaultFormats
      val gcMetadata = DefaultParamsReader.loadMetadata(path, sparkSession.sparkContext, className)
      val numClasses = (gcMetadata.metadata \ "numClasses").extract[Int]
      new BLBGCForestClassificationModel(gcMetadata.uid, Array[Array[GCForestClassificationModel]](), numClasses)
    }
  }
}



