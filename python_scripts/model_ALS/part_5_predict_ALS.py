"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark import SparkContext
from pyspark.sql.types import IntegerType, FloatType, DateType, StructType, StructField, StringType
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import time

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

# Load model
model = ALSModel.load(path="ml_models/my_als_model_2021-05-05_samlpe_20_percents")

n_recommendations = model.recommendForAllUsers(numItems=5)
n_recommendations.show(n=10)
