package com.test.sparkpoc;

/**
 * Hello world!
 *
 */
//Importing the necessary classes
//import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
//import org.apache.spark.api.java.JavaRDD;
//import org.apache.spark.SparkConf;
//import org.apache.spark.mllib.regression.LabeledPoint;
//import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
//import org.apache.spark.sql.Row;
//import org.apache.spark.mllib.classification.SVMModel;
//import org.apache.spark.mllib.classification.SVMWithSGD;
//import scala.Tuple2;
//import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.sql.SparkSession;

//Creating an Object earthquake
public class Earthquake {
	public static void main(String[] args) {
		//Creating a Spark Configuration and Spark Context
		//		SparkConf conf = new SparkConf().setAppName("Spark POC").setMaster("local");
		//		JavaSparkContext sparkContext = new JavaSparkContext(conf);

		//Loading the Earthquake ROC Dataset file as a LibSVM file
		//		val data = MLUtils.loadLibSVMFile(sc, *Path to the Earthquake File* )
		//		JavaRDD<LabeledPoint> data = 
		//				  MLUtils.loadLibSVMFile(sparkContext.sc(), "C:\\Users\\sathirem\\Desktop\\MyDocs\\Hadoop\\EarthQuakeROCDataset.xlsx").toJavaRDD();


		//Training the data for Machine Learning
		//	    JavaRDD<LabeledPoint> training = data.sample(false, 0.6, 11L);
		//	    training.cache();
		//	    JavaRDD<LabeledPoint> test = data.subtract(training);


		//Creating a model of the trained data
		//	    int numIterations = 100;
		//	    SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);

		//Using map transformation of model RDD
		//		val scoreAndLabels = *Map the model to predict features* 
		//	    JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(p ->
		//	      new Tuple2<>(model.predict(p.features()), p.label()));
		//Using Binary Classification Metrics on scoreAndLabels
		//		val metrics = * Use Binary Classification Metrics on scoreAndLabels *(scoreAndLabels)
		//		val auROC = metrics. *Get the area under the ROC Curve*()
		//	    BinaryClassificationMetrics metrics =
		//	    	      new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
		//	    	    double auROC = metrics.areaUnderROC();
		//Displaying the area under Receiver Operating Characteristic
		//		System.out.println("Area under ROC = " + auROC);


		// Save and load model
		//	    model.save(sparkContext, "target/tmp/javaSVMWithSGDModel");
		//	    SVMModel sameModel = SVMModel.load(sparkContext, "target/tmp/javaSVMWithSGDModel");
		// $example off$

		//	    sc.stop();

		String logFile = "C:\\\\Users\\\\sathirem\\\\Desktop\\\\MyPath.txt"; // Should be some file on your system
		SparkSession spark = SparkSession.builder().appName("Simple Application").getOrCreate();
		Dataset<String> logData = spark.read().textFile(logFile).cache();

		long numAs = logData.filter((FilterFunction<String>)s -> s.contains("a")).count();
		long numBs = logData.filter((FilterFunction<String>)s -> s.contains("b")).count();

		System.out.println("Lines with a: " + numAs + ", lines with b: " + numBs);

		spark.stop();
	}
}
