"""
/spark2.4/bin/pyspark --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.2
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.clustering import KMeans, KMeansModel

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Loading the model and data
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
w2v_model = Word2VecModel.load(path=user_path + 'ml_models/word2vec_model_2021_05_11')
kmeans_best_params = KMeans.load(path=user_path + 'ml_models/kmeans_2021-05-12')
kmeans_model = KMeansModel.load(path=user_path + 'ml_models/kmeans_model_2021-05-12')

product_vectors = w2v_model.getVectors().withColumnRenamed(existing='word', new='product_id')
products = spark \
    .read.format("org.apache.spark.sql.cassandra") \
    .options(table="products", keyspace="final_project").load() \
    .withColumn('name', F.regexp_replace('name', r'(\(\d+\) )', ''))

# Make predictions
predictions = kmeans_model.transform(product_vectors)


def show_products_of_one_cluster(num_cluster, n_rows, with_sort=True):
    print('\nNumber of  current cluser = ' + str(num_cluster))
    predictions_filtered = predictions \
        .where(condition=F.col('prediction') == num_cluster) \
        .select('product_id') \
        .join(other=products, on='product_id', how='left')
    predictions_filtered = predictions_filtered.orderBy('name', ascending=True) if with_sort else predictions_filtered
    return predictions_filtered.show(n=n_rows, truncate=False)


show_products_of_one_cluster(num_cluster=10, n_rows=15, with_sort=True)
"""
Number of  current cluser = 10
+----------+----------------------------------------------------------------+
|product_id|name                                                            |
+----------+----------------------------------------------------------------+
|115298    |Аир [корневища 1,5г фильтр-пакет уп] N20 КЛС 617                |
|112056    |Аир [корневища пачка 75г] N1 КЛС 617                            |
|142245    |Алтей [корень коробка 75г] N1 КЛС 617                           |
|107187    |Анис [плоды] 50г N1 617                                         |
|55079     |Багульник [болотного побеги пачка 50г] N1 КЛС 617               |
|35217     |Береза [лист пачка 50г] N1 617                                  |
|64809     |Береза [лист фильтр-пакет 1,5г] N20 КЛС 617                     |
+----------+----------------------------------------------------------------+
"""

for i in range(21):
    show_products_of_one_cluster(num_cluster=i, n_rows=6, with_sort=False)
