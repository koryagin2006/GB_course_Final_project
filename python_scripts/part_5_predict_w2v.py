"""
/spark2.4/bin/pyspark --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.2
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2VecModel

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"

# Loading the model and data
loadedModel = Word2VecModel.load(path='ml_models/word2vec_model_2021_05_11')
products = spark.read \
    .format("org.apache.spark.sql.cassandra").options(table="products", keyspace="final_project") \
    .load()


def get_synonyms_to_dataframe(model, word, num_synonyms=5):
    name = products.where(condition=F.col('product_id') == word).select('name').collect()[0]['name']
    print('\n similar for %s' % name)
    return model \
        .findSynonyms(word=str(word), num=num_synonyms) \
        .select('word').withColumnRenamed(existing='word', new='product_id') \
        .join(other=products, on='product_id', how='inner')


def get_synonyms_to_list(model, word, num_synonyms=5):
    recs_df = model \
        .findSynonyms(word=str(word), num=num_synonyms) \
        .select('word').withColumnRenamed(existing='word', new='product_id') \
        .join(other=products, on='product_id', how='inner')
    return [int(row.product_id) for row in recs_df.collect()]


required_product_id = 44530
get_synonyms_to_dataframe(model=loadedModel, word=required_product_id, num_synonyms=3).show(truncate=False)
"""
 similar for (3252) КЛОРАН Бэби вода очищающая мицеллярная 500мл 496
+----------+-------------------------------------------------------------------------+
|product_id|name                                                                     |
+----------+-------------------------------------------------------------------------+
|117626    |(67410) Нюкс Нирванеск Крем для контура глаз тюбик15 мл 501              |
|93501     |(66498) ДЮКРЭ Иктиан Молочко для тела увлажняющее 500 мл 549             |
|101228    |(98140) КЛОРАН Бэби Крем увлажняющий д/детей Витамины-Календула 200мл 772|
+----------+-------------------------------------------------------------------------+
"""

get_synonyms_to_list(model=loadedModel, word=required_product_id, num_synonyms=3)
""" [117626, 93501, 101228] """
