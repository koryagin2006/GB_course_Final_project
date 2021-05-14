"""
/spark2.4/bin/pyspark --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.2
https://habr.com/ru/company/jetinfosystems/blog/467745/
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import time

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


def get_silhouette_scores(vectors_df, features_col, clusters_list):
    start = time.time()
    evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol='vector', metricName='silhouette')
    silhouette_scores_dict = dict()
    for i in clusters_list:
        KMeans_algo = KMeans(featuresCol=features_col, k=i, maxIter=20, seed=3)
        KMeans_fit = KMeans_algo.fit(vectors_df)
        output = KMeans_fit.transform(vectors_df)
        score = evaluator.evaluate(output)
        print('i: {}, score: {}, time: {}'.format(i, score, str(time.time() - start)))
        silhouette_scores_dict[i] = score
    scores_df = spark.createDataFrame(data=list(map(list, silhouette_scores_dict.items())),
                                      schema=["n_clusters", "score"])
    return scores_df


scores_df = get_silhouette_scores(clusters_list=range(5, 200, 1),
                                  vectors_df=product_vectors,
                                  features_col='vector')
scores_df.orderBy('score', ascending=False).show(n=3)
"""
+----------+-------------------+
|n_clusters|              score|
+----------+-------------------+
|        21|0.24705539575632268|
|        17|0.23530894861309629|
|        12|0.22083229042257424|
+----------+-------------------+
"""

# Trains a k-means model. Best params
best_k = 21

kmeans_best_params = KMeans(featuresCol='vector', k=best_k, maxIter=20, seed=3)
kmeans_model = kmeans_best_params.fit(product_vectors)

# Save model
kmeans_best_params.save(path=user_path + 'ml_models/kmeans_2021-05-12')
kmeans_model.save(path=user_path + 'ml_models/kmeans_model_2021-05-12')
