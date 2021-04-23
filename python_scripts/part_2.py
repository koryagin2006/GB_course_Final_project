"""
/spark2.4/bin/pyspark \
    --driver-memory 512m \
    --driver-cores 1
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

# для начала готовим DataFrame
data = spark.read \
    .options(delimiter=',', inferschema=True, header=True) \
    .csv(path="input_csv_for_recommend_system/data.csv")
data.printSchema()
data.show(n=3, truncate=True)

