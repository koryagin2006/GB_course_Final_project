"""
/spark2.4/bin/pyspark --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.2
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Loading the model and data
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"

w2v_model = Word2VecModel.load(path=user_path + 'ml_models/word2vec_model_2021_05_11')
product_vectors = w2v_model.getVectors().withColumnRenamed(existing='word', new='product_id')
products = spark \
    .read.format("org.apache.spark.sql.cassandra") \
    .options(table="products", keyspace="final_project").load() \
    .withColumn('name', F.regexp_replace('name', r'(\(\d+\) )', ''))


# Trains a k-means models
def kmeans_model_fit(k, dataset):
    kmeans = KMeans(featuresCol='vector', maxIter=20, seed=3)
    kmeans_model = kmeans.fit(dataset=dataset, params={kmeans.k: k})
    predictions = kmeans_model.transform(product_vectors)
    return predictions


def show_products_of_one_cluster(num_cluster, predictions, with_sort=True, ):
    print('\nNumber of  current cluser = ' + str(num_cluster))
    predictions_filtered = predictions \
        .where(condition=F.col('prediction') == num_cluster) \
        .select('product_id') \
        .join(other=products, on='product_id', how='inner')
    predictions_filtered = predictions_filtered.orderBy('name', ascending=True) if with_sort else predictions_filtered
    return predictions_filtered


predictions_level_1 = kmeans_model_fit(k=21, dataset=product_vectors)
predictions_level_1.show()

# Отберем только какой-нибудь один кластер,например №3 - на вид - для беременных
show_products_of_one_cluster(num_cluster=3, predictions=predictions_level_1, with_sort=False).show(n=30, truncate=False)

product_vectors_lvl_2 = product_vectors \
    .join(other=predictions_level_1.select('product_id', 'prediction'), on='product_id', how='inner') \
    .where(F.col('prediction') == 3).select('product_id', 'vector')

predictions_level_2 = kmeans_model_fit(k=8, dataset=product_vectors_lvl_2)

for i in range(8):
    show_products_of_one_cluster(num_cluster=i, predictions=predictions_level_2, with_sort=True) \
        .show(n=15, truncate=False)
