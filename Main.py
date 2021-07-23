# Set up input and output files
input_data = "/databricks-datasets/credit-card-fraud/data"
output_test_parquet_data = "/tmp/pydata/credit-card-frauld-test-data"

data = spark.read.parquet(input_data)
display(data)

data.count()

from pyspark.ml.feature import  OneHotEncoder, VectorAssembler, VectorSizeHint
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline

from pyspark.sql.types import *
from pyspark.sql.functions import count, rand, collect_list, explode, struct, count
from pyspark.sql.functions import col
from pyspark.sql.functions import *

oneHot = OneHotEncoder(inputCols=["amountRange"], outputCols=["amountVect"])

# vectorSizeHint transformer used here because vectorAssembler can only work on 
# columns of known size
vectorSizeHint = VectorSizeHint(inputCol="pcaVector", size=28)

vectorAssembler = VectorAssembler(inputCols=["amountVect", "pcaVector"], outputCol="features")

estimator = GBTClassifier(labelCol="label", featuresCol="features")

# create the ML pipeline with four stages
pipeline = Pipeline(stages=[oneHot, vectorSizeHint, vectorAssembler, estimator])

# split training and test data
train = data.filter(col("time") % 10 < 8)
test = data.filter(col("time") % 10 >= 8)

# save our data into partitions so we can read them as files
(test.repartition(20).write
  .mode("overwrite")
  .parquet(output_test_parquet_data))

display(test)
test.count()

# fit the pipeline with the training dataset
pipelineModel = pipeline.fit(train)

# We would use the PySpark APIs to read off the file system to simulate as a stream from Kafka clusters
# define the schema
from pyspark.sql.types import *
from pyspark.ml.linalg import VectorUDT

schema = (StructType([StructField("time", IntegerType(), True), 
                      StructField("amountRange", IntegerType(), True), 
                      StructField("label", IntegerType(), True), 
                      StructField("pcaVector", VectorUDT(), True)]))

# simulate data streaming by reading one file at a time
streamingData = (spark.readStream 
                 .schema(schema) 
                 .option("maxFilesPerTrigger", 1) 
                 .parquet(output_test_parquet_data)
                 )

stream = pipelineModel.transform(streamingData)