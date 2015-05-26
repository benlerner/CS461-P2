package edu.drexel.cs461.apriori;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * 
 * @author  
 * Frequent itemset mining using Apache Spark SQL.
 * Ben Lerner
 * Madeline Harnos
 */
public final class Apriori {
    
    private static JavaSparkContext sparkContext;
    private static SQLContext sqlContext;
    
    /**
     * Set up Spark and SQL contexts.
     */
    private static void init (String master, int numReducers) {
	
	Logger.getRootLogger().setLevel(Level.WARN);
	
	SparkConf sparkConf = new SparkConf().setAppName("Apriori")
	    .setMaster(master)
	    .set("spark.sql.shuffle.partitions", "" + numReducers);
	
	sparkContext = new JavaSparkContext(sparkConf);
	sqlContext = new org.apache.spark.sql.SQLContext(sparkContext);
    }
    
    /**
     * 
     * @param inFileName
     * @return
     */
    private static DataFrame initXact (String inFileName) {
	
	// read in the transactions file
	JavaRDD<String> xactRDD = sparkContext.textFile(inFileName);
	
	// establish the schema: XACT (tid: string, item: int)
	List<StructField> fields = new ArrayList<StructField>();
	fields.add(DataTypes.createStructField("tid", DataTypes.StringType, true));
	fields.add(DataTypes.createStructField("item", DataTypes.IntegerType, true));
	StructType xactSchema = DataTypes.createStructType(fields);

	JavaRDD<Row> rowRDD = xactRDD.map(
					  new Function<String, Row>() {
					      static final long serialVersionUID = 42L;
					      public Row call(String record) throws Exception {
						  String[] fields = record.split("");
						  return  RowFactory.create(fields[0], Integer.parseInt(fields[1].trim()));
					      }
					  });

	// create DataFrame from xactRDD, with the specified schema
	return sqlContext.createDataFrame(rowRDD, xactSchema);
    }
    
    private static void saveOutput (DataFrame df, String outDir, String outFile) throws IOException {
	
	File outF = new File(outDir);
        outF.mkdirs();
        BufferedWriter outFP = new BufferedWriter(new FileWriter(outDir + "/" + outFile));
            
	List<Row> rows = df.toJavaRDD().collect();
	for (Row r : rows) {
	    outFP.write(r.toString() + "\n");
	}
        
        outFP.close();

    }
    
    public static void main(String[] args) throws Exception {

	if (args.length != 5) {
	    System.err.println("Usage: Apriori <inFile> <support> <outDir> <master> <numReducers>");
	    System.exit(1);
	}

	String inFileName = args[0].trim();
	double thresh =  Double.parseDouble(args[1].trim());
	String outDirName = args[2].trim();
	String master = args[3].trim();
	int numReducers = Integer.parseInt(args[4].trim());

	Apriori.init(master, numReducers);
	DataFrame xact = Apriori.initXact(inFileName);
	
	// compute frequent pairs (itemsets of size 2), output them to a file
	DataFrame frequentPairs = null;
	// your code goes here
	
	try {
	    Apriori.saveOutput(frequentPairs, outDirName + "/" + thresh, "pairs");
	} catch (IOException ioe) {
	    System.out.println("Cound not output pairs " + ioe.toString());
	}

	// compute frequent triples (itemsets of size 3), output them to a file
	DataFrame frequentTriples = null;
	// your code goes here
	
	try {
	    Apriori.saveOutput(frequentTriples, outDirName + "/" + thresh, "triples");
	} catch (IOException ioe) {
	    System.out.println("Cound not output triples " + ioe.toString());
	}
	
	sparkContext.stop();
	        
    }
}
