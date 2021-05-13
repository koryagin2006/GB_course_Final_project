# Построение модели коллаборативной фильтрации (Collaborative filtering)

https://spark.apache.org/docs/2.4.7/ml-collaborative-filtering.html

#### Запускаем spark приложение

```bash
/spark2.4/bin/pyspark
```

#### Импортируем необходимые библиотеки

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import time
```

### Подготовка

#### Загружаем данные о продажах в spark datafdame. Отберем только необходимые колонки, а также создадим дополнительный столбец с номером недели в году для последующего разбиения

```python
spark.sparkContext.setLogLevel("ERROR")
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"

# Load Data
data = spark.read.parquet(user_path + "input_csv_for_recommend_system/data.parquet")
data = data
    .select('sale_date_date', 'contact_id', 'product_id', 'quantity')
    .withColumn('quantity', F.when(F.col("quantity") != 1, 1).otherwise(F.col("quantity")))
    .withColumnRenamed(existing='product_id', new='item_id')
    .withColumnRenamed(existing='contact_id', new='user_id')
    .withColumn('week_of_year', F.weekofyear(F.col('sale_date_date')))
data.show(n=5)
```

```shell
+-------------------+-------+-------+--------+------------+
|     sale_date_date|user_id|item_id|quantity|week_of_year|
+-------------------+-------+-------+--------+------------+
|2018-12-07 00:00:00|1260627| 168308|     1.0|          49|
|2018-12-07 00:00:00| 198287| 134832|     1.0|          49|
|2018-12-07 00:00:00|2418385| 101384|     1.0|          49|
|2018-12-07 00:00:00|1285774| 168570|     1.0|          49|
|2018-12-07 00:00:00|1810323| 168319|     1.0|          49|
+-------------------+-------+-------+--------+------------+
```

#### Отберем только 15 последних недель для обучения, из-за ограниченности вычислительной мощности кластера

```python
def sample_by_week(df, week_col_name, split_size_weeks):
    threshold_week = int(data.select(F.max(week_col_name)).collect()[0][0]) - split_size_weeks
    df_before = df.filter(F.col(week_col_name) < threshold_week)
    df_after = df.filter(F.col(week_col_name) >= threshold_week)
    return df_before, df_after


before, data = sample_by_week(df=data, week_col_name='week_of_year', split_size_weeks=15)
data.orderBy('sale_date_date', ascending=True).show(n=3)
```

```shell
+-------------------+-------+-------+--------+------------+
|     sale_date_date|user_id|item_id|quantity|week_of_year|
+-------------------+-------+-------+--------+------------+
|2018-11-08 00:00:00|2591126|  32087|     1.0|          45|
|2018-11-08 00:00:00|2542992|  97117|     1.0|          45|
|2018-11-08 00:00:00|2477043| 106860|     1.0|          45|
+-------------------+-------+-------+--------+------------+
```

#### Просмотрим статистические данные по выборке

```python
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


basic_statistics_of_data().show(truncate=False)
```

```shell
+--------------------+------------+
|statistic           |value       |
+--------------------+------------+
|total number of rows|4'603'016   |
|number of users     |854'281     |
|number of items     |22'921      |
|sparsity            |99.97% empty|
+--------------------+------------+
```

#### Сделаем случайное разбиение на тестовую и обучающую выборки. Для теста возьмем 10%, т.к данных достаточно много

```python
(train, test) = data.randomSplit(weights=[0.9, 0.1], seed=3)
```

### Обучение модели

```python
als = ALS(userCol="user_id", itemCol="item_id", ratingCol="quantity",
          nonnegative=True, implicitPrefs=True, coldStartStrategy="drop")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="quantity", predictionCol="prediction")

start = time.time()
model_als = als.fit(train)
print('time = ' + str(time.time() - start))  
```

```shell
time = 52.6529769897
```

```python
# Save model_als
model_als.save(path=user_path + "ml_models/my_als_model_2021-05-11_last_15_weeks.model_als")
```

#### Параметры модели ALS

```python
spark.createDataFrame(
    data=[('Rank', str(model.rank)), ('MaxIter', str(als.getMaxIter())), ('RegParam', str(als.getRegParam()))],
    schema=['parameter', 'value']).show()
```

```shell
+---------+-----+
|parameter|value|
+---------+-----+
|     Rank|   10|
|  MaxIter|   10|
| RegParam|  0.1|
+---------+-----+
```

#### Сделаем прогноз на тестовой выборке

```python
test_predictions = model_als.transform(test)
test_predictions.show(n=5)
```

```shell
+-------------------+-------+-------+--------+------------+------------+
|     sale_date_date|user_id|item_id|quantity|week_of_year|  prediction|
+-------------------+-------+-------+--------+------------+------------+
|2018-11-16 00:00:00| 396523|   8086|     1.0|          46| 0.003291605|
|2018-11-11 00:00:00|1642159|   8086|     1.0|          45|0.0030388434|
|2018-11-17 00:00:00|2025608|   8086|     1.0|          46| 3.899849E-5|
|2018-11-24 00:00:00|1200425|   8086|     1.0|          47|1.8746716E-4|
|2018-11-15 00:00:00|1996289|   8086|     1.0|          46| 0.002490461|
+-------------------+-------+-------+--------+------------+------------+
```

# Метрики качества

#### Создадим результирующую таблицу с реальными и предсказанными товарами для оценки качества

```python
train_actual_items = train
    .select('user_id', 'item_id')
    .groupBy('user_id').agg(F.collect_list(col='item_id'))
    .withColumnRenamed(existing='collect_list(item_id)', new='actual')

train_recs_items = model.recommendForAllUsers(numItems=5)
    .select('user_id', F.col("recommendations.item_id").alias('recs_ALS'))

result = train_actual_items.join(other=train_recs_items, on='user_id', how='inner')
result.show(n=5, truncate=True)
```

```shell
+-------+--------------------+--------------------+
|user_id|              actual|            recs_ALS|
+-------+--------------------+--------------------+
|    463|     [102659, 66900]|[61115, 138005, 1...|
|    471|[51466, 28784, 28...|[162780, 135427, ...|
|   1238|     [59334, 102788]|[41096, 102788, 4...|
|   1342|     [97772, 110565]|[110629, 156491, ...|
|   1580|     [60809, 153583]|[138005, 61115, 1...|
+-------+--------------------+--------------------+
```

#### Выведем метрики оценки качества модели

```python
RMSE = evaluator.evaluate(test_predictions)
metrics = RankingMetrics(predictionAndLabels=result.select('actual', 'recs_ALS').rdd.map(tuple))
metrics_df = spark.createDataFrame(data=[('RMSE', RMSE),
                                         ('precision@k', metrics.precisionAt(5)),
                                         ('ndcg@k', metrics.ndcgAt(5)),
                                         ('meanAVGPrecision', metrics.meanAveragePrecision)],
                                   schema=['metric', 'value'])
metrics_df.withColumn('value', F.round('value', 5)).show(truncate=False)
```

```shell
+----------------+-------+
|metric          |value  |
+----------------+-------+
|RMSE            |0.98132|
|precision@k     |0.06178|
|ndcg@k          |0.06803|
|meanAVGPrecision|0.04082|
+----------------+-------+
```