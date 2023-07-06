/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.classification

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.CompletelyRandomForestImpl
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset


class CompletelyRandomForestClassifier(override val uid: String)
  extends RandomForestClassifier613 {

  def this() = this(Identifiable.randomUID("crfc"))

  override protected def train(dataset: Dataset[_]): RandomForestClassificationModel613 = instrumented { instr =>
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity)

//    instr.logParams(params: _*)
    instr.logParams(this, labelCol, featuresCol, predictionCol, probabilityCol, rawPredictionCol,
      impurity, numTrees, featureSubsetStrategy, maxDepth, maxBins, maxMemoryInMB, minInfoGain,
      minInstancesPerNode, seed, subsamplingRate, thresholds, cacheNodeIds, checkpointInterval)

    println("CompletelyRandomForestImplV0")
    val trees = CompletelyRandomForestImpl
      .run(oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeClassificationModel])

    val numFeatures = oldDataset.first().features.size
    val m = new RandomForestClassificationModel613(trees, numFeatures, numClasses)
//    instr.logSuccess(m)
    m
  }
}
