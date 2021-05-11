"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.types import IntegerType, FloatType, DateType, StructType, StructField, StringType
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
    .select('sale_date_date', 'contact_id', 'product_id', 'quantity') \
    .withColumn('quantity', F.when(F.col("quantity") != 1, 1).otherwise(F.col("quantity"))) \
    .withColumnRenamed(existing='product_id', new='item_id') \
    .withColumnRenamed(existing='contact_id', new='user_id') \
    .withColumn('week_of_year', F.weekofyear(F.col('sale_date_date')))


def sample_by_week(df, week_col_name, split_size_weeks):
    threshold_week = int(data.select(F.max(week_col_name)).collect()[0][0]) - split_size_weeks
    df_before = df.filter(F.col(week_col_name) < threshold_week)
    df_after = df.filter(F.col(week_col_name) >= threshold_week)
    return df_before, df_after


# Отберем только 15 последних недель для обучения
before, after = sample_by_week(df=data, week_col_name='week_of_year', split_size_weeks=15)
data = after


def basic_statistics_of_data():
    numerator = data.select("quantity").count()
    num_users, num_items = data.select("user_id").distinct().count(), data.select("item_id").distinct().count()
    denominator = num_users * num_items
    sparsity = (1.0 - (numerator * 1.0) / denominator) * 100
    return spark.createDataFrame(data=[('total number of rows', str('{0:,}'.format(numerator).replace(',', '\''))),
                                       ('number of users', str('{0:,}'.format(num_users).replace(',', '\''))),
                                       ('number of items', str('{0:,}'.format(num_items).replace(',', '\''))),
                                       ('sparsity', str(sparsity)[:5] + "% empty")],
                                 schema=['statistic', 'value'])


basic_statistics_of_data.show(truncate=False)
"""
+--------------------+------------+
|statistic           |value       |
+--------------------+------------+
|total number of rows|4'603'016   |
|number of users     |854'281     |
|number of items     |22'921      |
|sparsity            |99.97% empty|
+--------------------+------------+
"""
# Create test and train set
(train, test) = data.randomSplit(weights=[0.9, 0.1], seed=3)

# Create ALS model
als = ALS(userCol="user_id", itemCol="item_id", ratingCol="quantity",
          nonnegative=True, implicitPrefs=True, coldStartStrategy="drop")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="quantity", predictionCol="prediction")

start = time.time()
model = als.fit(train)
print('time = ' + str(time.time() - start))  # time = 52.6529769897

# Save model
# TODO: Не работает сохранение модели
model.save(path=user_path + "ml_models/my_als_model_2021-05-11_last_15_weeks.model")

# Параметры модели ALS.
spark.createDataFrame(
    data=[('Rank', str(model.rank)), ('MaxIter', str(als.getMaxIter())), ('RegParam', str(als.getRegParam()))],
    schema=['parameter', 'value']).show()

# Посмотрим прогнозы
test_predictions = model.transform(test)
RMSE = evaluator.evaluate(test_predictions)
print('RMSE = ' + str(round(RMSE, 4)))  # RMSE = 0.9813

# Создадим таблицу с реальными и предсказанными товарами
train_actual_items = train \
    .select('user_id', 'item_id') \
    .groupBy('user_id').agg(F.collect_list(col='item_id')) \
    .withColumnRenamed(existing='collect_list(item_id)', new='actual')

train_recs_items = model.recommendForAllUsers(numItems=5) \
    .select('user_id', F.col("recommendations.item_id").alias('recs_ALS'))

result = train_actual_items.join(other=train_recs_items, on='user_id', how='inner')
result.show(n=5, truncate=True)
"""
+-------+--------------------+--------------------+
|user_id|              actual|            recs_ALS|
+-------+--------------------+--------------------+
|    463|     [102659, 66900]|[61115, 138005, 1...|
|    471|[51466, 148601, 8...|[162780, 135427, ...|
|   1238|     [59334, 102788]|[41096, 102788, 4...|
|   1342|     [97772, 110565]|[110629, 156491, ...|
|   1580|     [60809, 153583]|[138005, 61115, 1...|
+-------+--------------------+--------------------+
"""
# Метрики качества
metrics = RankingMetrics(predictionAndLabels=result.select('actual', 'recs_ALS').rdd.map(tuple))
metrics_df = spark.createDataFrame(data=[('precision@k', metrics.precisionAt(5)),
                                         ('ndcg@k', metrics.ndcgAt(5)),
                                         ('meanAVGPrecision', metrics.meanAveragePrecision)],
                                   schema=['metric', 'value'])
metrics_df.withColumn('value', F.round('value', 5)).show(truncate=False)
"""
+----------------+-------+
|metric          |value  |
+----------------+-------+
|precision@k     |0.06201|
|ndcg@k          |0.06824|
|meanAVGPrecision|0.04092|
+----------------+-------+
"""

train_recs_final = model \
    .recommendForAllUsers(numItems=5) \
    .withColumn(colName="rec_exp", col=F.explode("recommendations")) \
    .select('user_id', F.col("rec_exp.item_id"), F.col("rec_exp.rating"))


def predict_als(user_id, n_recs=3, model=model):
    return model \
        .recommendForAllUsers(numItems=n_recs) \
        .where(condition=F.col('user_id') == user_id) \
        .withColumn(colName="rec_exp", col=F.explode("recommendations")) \
        .select('user_id', F.col("rec_exp.item_id"))


predict_als(user_id=471, n_recs=3).show()
"""
+-------+-------+
|user_id|item_id|
+-------+-------+
|    471| 162780|
|    471| 135427|
|    471|  46797|
+-------+-------+
"""
