# Построение модели Word2Vec

https://spark.apache.org/docs/2.4.7/ml-features.html#word2vec

#### Запускаем spark приложение

```bash
/spark2.4/bin/pyspark
```

#### Импортируем необходимые библиотеки

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql.types import DateType, StringType
import time
```

#### Загружаем данные о продажах в spark datafdame. Отберем только необходимые колонки

```python
spark.sparkContext.setLogLevel("ERROR")

# Load Data
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
data = spark.read.parquet(user_path + "input_csv_for_recommend_system/data.parquet")
```

## Подготовка данных

#### Необходимо преобразовать `contact_id` в StringType, а `sale_date_date` в DateType

```python
data = data
.select('sale_date_date', 'contact_id', 'shop_id', 'product_id', 'quantity')
.withColumn(colName="sale_date_date", col=data["sale_date_date"].cast(DateType()))
.withColumn(colName="product_id", col=data["product_id"].cast(StringType()))
data.show(n=5, truncate=True)
```

```shell
+--------------+----------+-------+----------+--------+
|sale_date_date|contact_id|shop_id|product_id|quantity|
+--------------+----------+-------+----------+--------+
|    2018-12-07|   1260627|   1455|    168308|     1.0|
|    2018-12-07|    198287|    279|    134832|     1.0|
|    2018-12-07|   2418385|    848|    101384|     1.0|
|    2018-12-07|   1285774|   1511|    168570|     1.0|
|    2018-12-07|   1810323|   1501|    168319|     1.0|
+--------------+----------+-------+----------+--------+
```

##### Просмотрим количество уникальных пользователей в нашем наборе данных

```python
users = data.select('contact_id').distinct()
print('Number of unique users = ' + str('{0:,}'.format(users.count()).replace(',', '\'')))
```

```shell
Number of unique users = 1'636'831
```

#### Сделаем разбиение на тестовую и обучающую выборки. Разбиение будем делать по-клиентам

Для теста возьмем 10% клиентов и сформируем выборки на основе принадлежности клиента к тесту или трейну

```python
(users_train, users_valid) = users.randomSplit(weights=[0.9, 0.1], seed=5)
```

```shell
num_train_users = 1473217
num_test_users = 163614
```

```python
train_df = data.join(other=users_train, on='contact_id', how='inner')
validation_df = data.join(other=users_valid, on='contact_id', how='inner')

print('train_df.count = {}\nvalidation_df.count = {}'.format(train_df.count(), validation_df.count()))
```

```shell
train_df.count = 17398192
validation_df.count = 1948492
```

#### Введем колонку, определяющую номер чека и уберем лишние

```python
def create_col_orders(df):
    return df
    .select(F.concat_ws('_', data.sale_date_date, data.shop_id, data.contact_id).alias('order_id'),
            'product_id', 'quantity')
    .groupBy('order_id')
    .agg(F.collect_list(col='product_id'))
    .withColumnRenamed(existing='collect_list(product_id)', new='actual_products')


train_orders = create_col_orders(df=train_df)
validation_orders = create_col_orders(df=validation_df)

train_orders.show(n=5)
```

```shell
+--------------------+--------------------+
|            order_id|     actual_products|
+--------------------+--------------------+
|2018-01-01 00:00:...|     [77808, 130823]|
|2018-01-01 00:00:...|     [60367, 125733]|
|2018-01-01 00:00:...|    [110629, 138956]|
|2018-01-01 00:00:...|[217227, 136540, ...|
|2018-01-01 00:00:...|[70951, 94613, 23...|
+--------------------+--------------------+
```

## Обучение модели

```python
word2Vec = Word2Vec(
    vectorSize=100, minCount=5, numPartitions=1, seed=33, windowSize=3,
    inputCol='actual_products', outputCol='result')

start = time.time()
model = word2Vec.fit(dataset=train_orders)
print('time = ' + str(time.time() - start))
```

#### Сохранение модели на hdfs

```python
model.save(path=user_path + 'ml_models/word2vec_model_2021_05_11')
```

Проверим, что модель сохранена успешно

```python
loadedModel = Word2VecModel.load(path='ml_models/word2vec_model_2021_05_11')
print('Good saving? -> ' + str(loadedModel.getVectors().first().word == model.getVectors().first().word))
```

```shell
Good saving? -> True
```

## Метрики

К сожалению для данной модели без онлайн потока данных о реальных продажах изменить качество не представляется
возможным.