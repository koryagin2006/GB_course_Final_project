# Построение модели кластеризации К-means

- https://habr.com/ru/company/jetinfosystems/blog/467745/
- https://spark.apache.org/docs/2.4.7/ml-clustering.html#k-means

#### Запускаем spark приложение

```bash
/spark2.4/bin/pyspark  \
    --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.2
```

#### Импортируем необходимые библиотеки

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import time

spark.sparkContext.setLogLevel("ERROR")
```


#### Загружаем данные о продажах в spark datafdame. Отберем только необходимые колонки
```python
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"

w2v_model = Word2VecModel.load(path=user_path + 'ml_models/word2vec_model_2021_05_11')
product_vectors = w2v_model.getVectors().withColumnRenamed(existing='word', new='product_id')
products = spark \
    .read.format("org.apache.spark.sql.cassandra") \
    .options(table="products", keyspace="final_project").load() \
    .withColumn('name', F.regexp_replace('name', r'(\(\d+\) )', ''))

product_vectors.show(n=5)
```
```shell
+----------+--------------------+
|product_id|              vector|
+----------+--------------------+
|    144322|[-0.0024441950954...|
|     58451|[-3.9240214391611...|
|     75120|[-0.0589764676988...|
|    153532|[-0.0256759468466...|
|    134530|[-0.0764870494604...|
+----------+--------------------+
```
```python
products.show(n=5, truncate=False)
```
```shell
+----------+----------------------------------------------------------+
|product_id|name                                                      |
+----------+----------------------------------------------------------+
|104124    |Раствор Ликосол-2000 для конт.линз фл 240мл 817           |
|92248     |Риностоп спрей наз. 0,05% фл. 15мл 701                    |
|350363    |Кларитросин табл.п.п.о. 500мг №10 403                     |
|129004    |Флуконазол-Тева капс. 150мг №1 622                        |
|125915    |Валсартан-Гидрохлоротиазид таб. п.п.о. 80мг+12,5мг №28 738|
+----------+----------------------------------------------------------+
```

## Подбор количества классов

Подбор осуществляется по максимальному значению коэффициента `silhouette`.

Коэффициент «силуэт» вычисляется с помощью среднего внутрикластерного расстояния (a) и среднего расстояния до ближайшего кластера (b) по каждому образцу. Силуэт вычисляется как (b - a) / max(a, b). Поясню: b — это расстояние между a и ближайшим кластером, в который a не входит. Можно вычислить среднее значение силуэта по всем образцам и использовать его как метрику для оценки количества кластеров.

Для вычисления используем функцию, в которую передаем список чисел кластеров

```python
def get_silhouette_scores(vectors_df, features_col, clusters_list):
    start = time.time()
    evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol='vector', metricName='silhouette')
    silhouette_scores_dict = dict()
    for i in clusters_list:
        KMeans_algo = KMeans(featuresCol=features_col, k=i, maxIter=20, seed=3)
        KMeans_fit = KMeans_algo.fit(vectors_df)
        output = KMeans_fit.transform(vectors_df)
        score = evaluator.evaluate(output)
        print('i: {}, score: {}, time: {}'. format(i, score, str(time.time() - start)))
        silhouette_scores_dict[i] = score
    scores_df = spark.createDataFrame(data=list(map(list, silhouette_scores_dict.items())),
                                      schema=["n_clusters", "score"])
    return scores_df
```

Побдор сделаем для чисел кластеров от 5 до 99 и 
```python
scores_df = get_silhouette_scores(clusters_list=range(5, 200, 1), 
                                  vectors_df=product_vectors, 
                                  features_col='vector')
scores_df \
    .orderBy('score', ascending=False) \
    .show(n=5)
```

```shell
+----------+-------------------+
|n_clusters|              score|
+----------+-------------------+
|        21|0.24705539575632268|
|        17|0.23530894861309629|
|        12|0.22083229042257424|
|        11|0.21774700055303492|
|        13|0.21705090230733062|
+----------+-------------------+
```

## Обучение конечной модели

По значению метрики наилучшее разбиение получается на 21 класс

```python
best_k = 21

kmeans_best_params = KMeans(featuresCol='vector', k=best_k, maxIter=20, seed=3)
kmeans_model = kmeans_best_params.fit(product_vectors)
```

#### Сохранение модели на hdfs

```python
kmeans_best_params.save(path=user_path + 'ml_models/kmeans_2021-05-12')
kmeans_model.save(path=user_path + 'ml_models/kmeans_model_2021-05-12')
```

```shell
393     786     ml_models/kmeans_2021-05-12
18.6 K  37.1 K  ml_models/kmeans_model_2021-05-12
7.5 M   15.0 M  ml_models/word2vec_model_2021_05_11
```
