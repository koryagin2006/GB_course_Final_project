"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.sql.types import IntegerType, FloatType, DateType, StructType, StructField, StringType
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import time

sc = SparkContext
# sc.setCheckpointDir('checkpoint')
spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"

# Create DF
data = spark.read.parquet(user_path + "input_csv_for_recommend_system/data.parquet")
data = data \
    .select('contact_id', 'product_id', 'quantity') \
    .withColumn('quantity', F.when(F.col("quantity") != 1, 1).otherwise(F.col("quantity"))) \
    .withColumnRenamed(existing='product_id', new='item_id') \
    .withColumnRenamed(existing='contact_id', new='user_id')

# Sample 20% of data
data = data.sample(fraction=0.2, seed=5)

# Basic statistics of data
numerator = data.select("quantity").count()
num_users = data.select("user_id").distinct().count()
num_items = data.select("item_id").distinct().count()
denominator = num_users * num_items
sparsity = (1.0 - (numerator * 1.0) / denominator) * 100

statistics = spark.createDataFrame(data=[('total number of rows', str('{0:,}'.format(numerator).replace(',', '\''))),
                                         ('number of users', str('{0:,}'.format(num_users).replace(',', '\''))),
                                         ('number of items', str('{0:,}'.format(num_items).replace(',', '\''))),
                                         ('sparsity', str(sparsity)[:5] + "% empty")],
                                   schema=StructType([StructField("statistic", StringType()),
                                                      StructField("value", StringType())]))
statistics.show(truncate=False)

# Create test and train set
(train, test) = data.randomSplit(weights=[0.9, 0.1], seed=3)

# Create ALS model
als = ALS(userCol="user_id", itemCol="item_id", ratingCol="quantity",
          nonnegative=True, implicitPrefs=True, coldStartStrategy="drop")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="quantity", predictionCol="prediction")

start = time.time()
model = als.fit(train)
end = time.time()

print('time = ' + str(end - start))

# Save model
# TODO: Не работает сохранение модели
model.save(user_path + "ml_models/my_als_model_2021-05-05_samlpe_20_percents")

# Complete the code below to extract the ALS model parameters
spark.createDataFrame(
    data=[('Rank', str(model.rank)), ('MaxIter', str(als.getMaxIter())), ('RegParam', str(als.getRegParam()))],
    schema=StructType([StructField("parameter", StringType()), StructField("value", StringType())])).show()

spark.createDataFrame(data=[('Rank', str(model.rank)),
                            ('MaxIter', str(als.getMaxIter())),
                            ('RegParam', str(als.getRegParam()))]).show()

# View the predictions
test_predictions = model.transform(test)
RMSE = evaluator.evaluate(test_predictions)
print('RMSE = ' + str(round(RMSE, 4)))

n_recommendations = model.recommendForAllUsers(numItems=5)
n_recommendations.show(n=10)

n_recommendations = n_recommendations \
    .withColumn(colName="rec_exp", col=F.explode("recommendations")) \
    .select('user_id', F.col("rec_exp.item_id"), F.col("rec_exp.rating"))
n_recommendations.show(n=10)

# TODO: Настроить нормальную метрику, типа map@k
# TODO: Настроить перевзвешивание tf или брать sum(quantity) / max(sum(quantity) over users)
