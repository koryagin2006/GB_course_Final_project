"""
/spark2.4/bin/pyspark --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.2
https://habr.com/ru/company/jetinfosystems/blog/467745/
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

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


def show_products_of_one_cluster(num_cluster, n_rows):
    print('\nNumber of  current cluser = ' + str(num_cluster))
    return predictions \
        .where(condition=F.col('prediction') == num_cluster) \
        .select('product_id') \
        .join(other=products, on='product_id', how='left') \
        .orderBy('name', ascending=True) \
        .show(n=n_rows, truncate=False)


show_products_of_one_cluster(num_cluster=19, n_rows=30)
show_products_of_one_cluster(num_cluster=10, n_rows=15)

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
