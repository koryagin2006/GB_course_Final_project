from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

