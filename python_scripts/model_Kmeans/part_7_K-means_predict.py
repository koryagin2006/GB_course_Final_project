"""
/spark2.4/bin/pyspark --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.2
"""

"""
https://habr.com/ru/company/jetinfosystems/blog/467745/
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pprint import pprint
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, DoubleType, StringType
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

# Loading the model and data
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
w2v_model = Word2VecModel.load(path=user_path + 'ml_models/word2vec_model_2021_05_11')
product_vectors = w2v_model.getVectors().withColumnRenamed(existing='word', new='product_id')
products = spark \
	.read.format("org.apache.spark.sql.cassandra") \
	.options(table="products", keyspace="final_project").load() \
	.withColumn('name', F.regexp_replace('name', r'(\(\d+\) )', ''))


# Trains a k-means model. Best params
best_k = 21
# best_k = scores_df.orderBy('score', ascending=False).head(n=1)[0]['n_clusters']
kmeans_best_params = KMeans(featuresCol='vector', k=best_k, maxIter=20, seed=3)
kmeans_model = kmeans_best_params.fit(product_vectors)

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


# for i in range(21):
# 	show_products_of_one_cluster(num_cluster=i, n_rows=10)
# 	print('\n')


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
|70279     |Березовые почки [коробка 50г] N1 КЛС 617                        |
|73624     |Бессмертник [песчаный цветки коробка 30г ] N1 КЛС 617           |
|74360     |Боярышник [плоды пачка 75г] N1 КЛС 617                          |
|138310    |Боярышник [плоды фильтр-пакеты 3г] №20 КЛС 617                  |
|151189    |Брусника (листья пачка 50г) N1 КЛС 617                          |
|32800     |Валериана [корневища с корнями 1,5г фильтр-пакет уп] N20 КЛС 617|
|76277     |Валериана [корневища с корнями пачка 50г] N1 КЛС 617            |
|150948    |Горец птичий (Спорыш) (сырье коробка 50г] N1 КЛС 617            |
+----------+----------------------------------------------------------------+

"""