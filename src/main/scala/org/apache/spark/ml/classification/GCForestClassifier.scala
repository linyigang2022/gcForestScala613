package org.apache.spark.ml.classification

import org.apache.hadoop.fs.Path
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.GCForestParams
import org.apache.spark.ml.tree.configuration.GCForestStrategy
import org.apache.spark.ml.tree.impl.GCForestImpl
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

class GCForestClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, GCForestClassifier, GCForestClassificationModel]
    with DefaultParamsWritable with GCForestParams {

  def this() = this(Identifiable.randomUID("gcf"))

  override def setNumClasses(value: Int): this.type = set(classNum, value)

  override def setModelPath(value: String): GCForestClassifier.this.type = set(modelPath, value)

  override def setDataSize(value: Array[Int]): this.type = set(dataSize, value)

  override def setDataStyle(value: String): this.type = set(dataStyle, value)

  override def setMultiScanWindow(value: Array[Int]): this.type = set(multiScanWindow, value)

  override def setScanForestTreeNum(value: Int): this.type = set(scanForestTreeNum, value)

  override def setCascadeForestTreeNum(value: Int): this.type = set(cascadeForestTreeNum, value)

  override def setMaxIteration(value: Int): this.type = set(MaxIteration, value)

  override def setEarlyStoppingRounds(value: Int): this.type = set(earlyStoppingRounds, value)

  override def setIDebug(value: Boolean): GCForestClassifier.this.type = set(idebug, value)

  override def setMaxDepth(value: Int): this.type = set(MaxDepth, value)

  override def setMaxBins(value: Int): GCForestClassifier.this.type = set(MaxBins, value)

  override def setMinInfoGain(value: Double): GCForestClassifier.this.type = set(minInfoGain, value)

  override def setScanForestMinInstancesPerNode(value: Int):
  GCForestClassifier.this.type = set(scanMinInsPerNode, value)

  override def setCascadeForestMinInstancesPerNode(value: Int):
  GCForestClassifier.this.type = set(cascadeMinInsPerNode, value)

  override def setFeatureSubsetStrategy(value: String):
  GCForestClassifier.this.type = set(featureSubsetStrategy, value)

  override def setCacheNodeId(value: Boolean):
  GCForestClassifier.this.type = set(cacheNodeId, value)

  override def setMaxMemoryInMB(value: Int):
  GCForestClassifier.this.type = set(maxMemoryInMB, value)

  override def setRFNum(value: Int): GCForestClassifier.this.type = set(rfNum, value)

  override def setCRFNum(value: Int): GCForestClassifier.this.type = set(crfNum, value)

  def getGCForestStrategy: GCForestStrategy = {
    GCForestStrategy($(classNum), $(modelPath), $(multiScanWindow),
      $(dataSize), $(rfNum), $(crfNum),
      $(scanForestTreeNum), $(cascadeForestTreeNum), $(scanMinInsPerNode),
      $(cascadeMinInsPerNode), $(featureSubsetStrategy), $(crf_featureSubsetStrategy), $(MaxBins),
      $(MaxDepth), $(minInfoGain), $(MaxIteration), $(maxMemoryInMB),
      $(numFolds), $(earlyStoppingRounds),
      $(earlyStopByTest), $(dataStyle), $(seed), $(cacheNodeId),
      $(windowCol), $(scanCol), $(forestIdCol), $(idebug))
  }

  def getDefaultStrategy: GCForestStrategy = {
    GCForestStrategy(2, $(modelPath), Array(), Array(113), idebug = false)
  }

  def train(trainset: Dataset[_], testset: Dataset[_]): GCForestClassificationModel = {
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

    copyValues(GCForestImpl.runWithValidation(casted_train, casted_test, getGCForestStrategy))
  }

  override def train(dataset: Dataset[_]): GCForestClassificationModel = {
    // This handles a few items such as schema validation.
    // Developers only need to implement train().
    transformSchema(dataset.schema, logging = true)

    // Cast LabelCol to DoubleType and keep the metadata.
    val labelMeta = dataset.schema($(labelCol)).metadata
    val casted_train =
      dataset.withColumn($(labelCol), col($(labelCol)).cast(DoubleType), labelMeta)
    GCForestImpl.run(casted_train, getGCForestStrategy)
  }

  override def copy(extra: ParamMap): GCForestClassifier = defaultCopy(extra)
}


private[ml] class GCForestClassificationModel(
                                               override val uid: String,
                                               private val scanModel: Array[MultiGrainedScanModel],
                                               private val cascadeForest: Array[Array[RandomForestClassificationModel]],
                                               override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, GCForestClassificationModel]
    with GCForestParams with MLWritable with Serializable {

  def this(scanModel: Array[MultiGrainedScanModel],
           cascadeForest: Array[Array[RandomForestClassificationModel]],
           numClasses: Int) =
    this(Identifiable.randomUID("gcfc"), scanModel, cascadeForest, numClasses)

  val numScans: Int = scanModel.length
  val numCascades: Int = cascadeForest.length

  def predictScanFeature(features: Vector): Vector = {
    features
  }

  override def predictRaw(features: Vector): Vector = features

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case _: SparseVector =>
        throw new RuntimeException("Unexpected error in GCForestClassificationModel:" +
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

  override def copy(extra: ParamMap): GCForestClassificationModel = {
    copyValues(new GCForestClassificationModel(uid, scanModel, cascadeForest, numClasses), extra)
  }

  override def write: MLWriter =
    new GCForestClassificationModel.GCForestClassificationModelWriter(this)
}


object GCForestClassificationModel extends MLReadable[GCForestClassificationModel] {
  override def read: MLReader[GCForestClassificationModel] = new GCForestClassificationModelReader

  override def load(path: String): GCForestClassificationModel = super.load(path)

  private[GCForestClassificationModel]
  class GCForestClassificationModelWriter(instance: GCForestClassificationModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {

      val gcMetadata: JObject = Map(
        "numClasses" -> instance.numClasses,
        "numScans" -> instance.numScans,
        "numCascades" -> instance.numCascades)
      DefaultParamsWriter.saveMetadata(instance, path, sparkSession.sparkContext, Some(gcMetadata))

      // scanModel
      val scanPath = new Path(path, "scan").toString
      instance.scanModel.zipWithIndex.foreach { case (model, index) =>
        val modelPath = new Path(scanPath, index.toString).toString
        val metadata: JObject = Map(
          "windows" -> model.windows.toList
        )
        DefaultParamsWriter
          .saveMetadata(model, modelPath, sparkSession.sparkContext, Some(metadata))
        val rfModelPath = new Path(modelPath, "rf").toString
        model.rfcModel.save(rfModelPath)
        val crtfModelPath = new Path(modelPath, "crtf").toString
        model.crfcModel.save(crtfModelPath)
      }

      // CascadeForestModel
      val cascadePath = new Path(path, "cascade").toString
      instance.cascadeForest.zipWithIndex.foreach { case (models, level) =>
        val modelsPath = new Path(cascadePath, level.toString).toString
        models.zipWithIndex.foreach { case (model, index) =>
          val modelPath = new Path(modelsPath, index.toString).toString
          model.save(modelPath)
        }
      }
    }
  }

  private class GCForestClassificationModelReader
    extends MLReader[GCForestClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GCForestClassificationModel].getName
    val mgsClassName: String = classOf[MultiGrainedScanModel].getName

    override def load(path: String): GCForestClassificationModel = {
      implicit val format = DefaultFormats
      val gcMetadata = DefaultParamsReader.loadMetadata(path, sparkSession.sparkContext, className)

      val numClasses = (gcMetadata.metadata \ "numClasses").extract[Int]
      val numScans = (gcMetadata.metadata \ "numScans").extract[Int]
      val numCascades = (gcMetadata.metadata \ "numCascades").extract[Int]

      val scanPath = new Path(path, "scan").toString
      val scanModel = Range(0, numScans).map { index =>
        val modelPath = new Path(scanPath, index.toString).toString
        val scanMetadata = DefaultParamsReader
          .loadMetadata(path, sparkSession.sparkContext, mgsClassName)
        val windows = (scanMetadata.metadata \ "windows").extract[Array[Int]]
        val rfPath = new Path(modelPath, "rf").toString
        val rfModel = RandomForestClassificationModel.load(rfPath)
        val crtfPath = new Path(modelPath, "crtf").toString
        val crtfModel = RandomForestClassificationModel.load(crtfPath)
        new MultiGrainedScanModel(windows, rfModel, crtfModel)
      }.toArray

      val cascadePath = new Path(path, "cascade").toString
      val cascadeForest = Range(0, numCascades).map { level =>
        val modelsPath = new Path(cascadePath, level.toString).toString
        Range(0, 4).map { index =>
          val modelPath = new Path(modelsPath, index.toString).toString
          RandomForestClassificationModel.load(modelPath)
        }.toArray
      }.toArray

      val gcForestModel =
        new GCForestClassificationModel(gcMetadata.uid, scanModel, cascadeForest, numClasses)

      //      DefaultParamsReader.getAndSetParams(gcForestModel, gcMetadata)
      gcMetadata.getAndSetParams(gcForestModel)
      gcForestModel
    }
  }
}

class MultiGrainedScanModel(override val uid: String,
                            val windows: Array[Int],
                            val rfcModel: RandomForestClassificationModel,
                            val crfcModel: RandomForestClassificationModel) extends Params {
  def this(windows: Array[Int],
           rfcModel: RandomForestClassificationModel,
           crfcModel: RandomForestClassificationModel) =
    this(Identifiable.randomUID("mgs"), windows, rfcModel, crfcModel)

  override def copy(extra: ParamMap): Params =
    copyValues(new MultiGrainedScanModel(uid, windows, rfcModel, crfcModel), extra)
}

