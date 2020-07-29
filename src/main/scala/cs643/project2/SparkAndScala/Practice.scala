package cs643.project2.SparkAndScala

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.{ LogisticRegressionModel, LogisticRegressionWithLBFGS }
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.hadoop.conf.Configuration

object Practice {

  def main(args: Array[String]): Unit = {

    //Logger.getLogger("org").setLevel(Level.ERROR)

    //val inputFile = args

    val spark = new SparkConf()
      /* SparkSession
      .builder()*/
      .setAppName("LogisticRegressionApp")
      .setMaster("local[4]")
      //.config("spark.driver.memory","4g")
      //.getOrCreate()
      //spark.conf.set("spark.sql.shuffle.partitions", 12)
      //spark.conf.set("spark.executor.memory", "8g")
      //spark.sparkContext.setLogLevel("ERROR")
      .set("spark.testing.memory", "2147480000")

    val sparkContext = new SparkContext(spark)

    sparkContext.setLogLevel("ERROR")

    val hadoopConfig: Configuration = sparkContext.hadoopConfiguration
    hadoopConfig.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
    hadoopConfig.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)

    val csv = sparkContext.textFile(args(0))

    val skip = csv.first()
    val csv_row = csv.filter(_(0) != skip(0))//row => row != skip)
    
    val data = csv_row.map(_.split(";")).map(x => (x(11).toInt, Vectors.dense(x(0).toDouble, x(1).toDouble, x(2).toDouble,
      x(3).toDouble, x(4).toDouble, x(5).toDouble, x(6).toDouble, x(7).toDouble, x(8).toDouble, x(9).toDouble,
      x(10).toDouble)))

    /*    import spark.implicits._
    val columnNames = Seq("label", "features")
    val df = data.toDF(columnNames: _*)
*/
    val labeledPointRDD = data map {
      case (label, features) => LabeledPoint(label, features)
    }

    val trainingTest = labeledPointRDD.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = trainingTest(0).cache()
    val test = trainingTest(1)

    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(training)

    val predictionAndLabels = test.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    //    val accuracy = metrics.accuracy
    val fScore = metrics.fMeasure

    //    println(s"Accuracy = $accuracy")
    println(s"F1 Score = $fScore")

  }

}