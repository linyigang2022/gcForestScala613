/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */

package org.apache.spark.ml.datasets

class BaseDatasets {

}
class FeatureParser(row: String) extends Serializable {
  private val desc = row.trim
  private val f_type = if (desc == "C") "number" else "categorical"
  private val name_to_len = if (f_type == "categorical") {
    val f_names = Array("?") ++ desc.trim.split(",").map(str => str.trim)
    f_names.zipWithIndex.map { case(cate_name, idx) =>
      cate_name -> idx
    }.toMap
  } else Map[String, Int]()

  def get_double(f_data: String): Double = {
    if (f_type == "number") f_data.trim.toDouble
    else name_to_len.getOrElse(f_data.trim, 0).toDouble
  }

  def get_data(f_data: String): Array[Double] = {
    if (f_type == "number") Array[Double](f_data.trim.toDouble)
    else {
      val data = Array.fill[Double](name_to_len.size)(0f)
      data(name_to_len.getOrElse(f_data.trim, 0)) = 1f
      data
    }
  }

  def get_fdim(): Int = {
    if (f_type == "number") 1 else name_to_len.size
  }
}