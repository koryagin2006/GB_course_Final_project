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


sc = SparkContext
# sc.setCheckpointDir('checkpoint')
spark = SparkSession.builder.appName('305_Recommendations').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


data = spark.read.parquet("hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/input_csv_for_recommend_system/data.parquet")
data = data \
    .select('contact_id', 'product_id', 'quantity') \
    .withColumn('quantity', F.when(F.col("quantity") != 1, 1).otherwise(F.col("quantity"))) \
    .withColumnRenamed(existing='product_id', new='item_id') \
    .withColumnRenamed(existing='contact_id', new='user_id')

# Сделаем сэмпл. Обучим на части датасета
data = data.sample(fraction=0.2, seed=5)


numerator = data.select("quantity").count()
num_users = data.select("user_id").distinct().count()
num_items = data.select("item_id").distinct().count()
denominator = num_users * num_items
sparsity = (1.0 - (numerator * 1.0) / denominator) * 100

df = spark.createDataFrame(data=[('total number of data', str('{0:,}'.format(numerator).replace(',', '\''))),
                                 ('number of users', str('{0:,}'.format(num_users).replace(',', '\''))),
                                 ('number of items', str('{0:,}'.format(num_items).replace(',', '\''))),
                                 ('sparsity', str(sparsity)[:5]+"% empty")],
                           schema=StructType([StructField("featute",StringType()), 
                                              StructField("value",StringType())]))
df.show(truncate=False)

# Create test and train set
(train, test) = data.randomSplit([0.9, 0.1], seed=3)

# Create ALS model
als = ALS(userCol="user_id", itemCol="item_id", ratingCol="quantity", 
          nonnegative=True, implicitPrefs=True, coldStartStrategy="drop")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="quantity", predictionCol="prediction")

t_start = time.time()

model = als.fit(train)

t_end = time.time()
print('time', t_end - t_start)

# Save model
model.save("ml_models/my_als_2021-05-05_samlpe_20_percents")

# Complete the code below to extract the ALS model parameters
print("Model")
print("     Rank:", model._java_obj.parent().getRank())
print("     MaxIter:", model._java_obj.parent().getMaxIter())
print("     RegParam:", model._java_obj.parent().getRegParam())

# View the predictions
test_predictions = model.transform(test)
RMSE = evaluator.evaluate(test_predictions)
print('RMSE = ', RMSE)

nrecommendations = model.recommendForAllUsers(5)
nrecommendations.show(n=10)

nrecommendations = nrecommendations \
    .withColumn(colName="rec_exp", col=explode("recommendations"))\
    .select('user_id', col("rec_exp.item_id"), col("rec_exp.rating"))
nrecommendations.show(n=10)



data.count()
data.printSchema()
data.show(n=10)
data.select('quantity').describe().show()
