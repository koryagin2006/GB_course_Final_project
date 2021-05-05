"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import col, explode
from pyspark import SparkContext
from pyspark.sql.types import IntegerType, FloatType, DateType, StructType, StructField, StringType

from pyspark.sql import SparkSession

# Import the required functions
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import functions as F

import time


spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

# Load model
model = ALSModel.load(path="ml_models/my_als_2021-05-05")