name := "SparkAndScala"

version := "1.0"

scalaVersion := "2.11.11"
val sparkVersion = "2.4.6"

libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-streaming" % sparkVersion

mainClass in (Compile, packageBin) := Some("cs643.project2.SparkAndScala.Practice")

assemblyMergeStrategy in assembly := {
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case x => MergeStrategy.first
}