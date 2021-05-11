"""
/spark2.4/bin/pyspark --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.2
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2VecModel

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

# Loading the model and data
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
loadedModel = Word2VecModel.load(path=user_path + 'ml_models/word2vec_model_2021_05_11')
products = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="products", keyspace="final_project") \
    .load()


def get_synonyms_to_dataframe(model, product_id, num_synonyms=5):
    name = products.where(condition=F.col('product_id') == product_id).select('name').collect()[0]['name']
    print('\n similar for %s' % name)
    return model \
        .findSynonyms(word=str(product_id), num=num_synonyms) \
        .withColumnRenamed(existing='word', new='product_id') \
        .join(other=products, on='product_id', how='inner') \
        .orderBy('similarity', ascending=False).withColumn('similarity', F.round('similarity', 6))


def get_synonyms_to_list(model, product_id, num_synonyms=5):
    recs_df = model \
        .findSynonyms(word=str(product_id), num=num_synonyms) \
        .withColumnRenamed(existing='word', new='product_id') \
        .join(other=products, on='product_id', how='inner') \
        .orderBy('similarity', ascending=False)
    return [int(row.product_id) for row in recs_df.collect()]


def view_product_list(n_rows):
    """ Вывести список товаров """
    return spark \
        .read.parquet(user_path + "input_csv_for_recommend_system/data.parquet") \
        .select('product_id').distinct() \
        .join(other=products, on='product_id', how='inner') \
        .show(n=n_rows, truncate=False)


# view_product_list(n_rows=30)

required_product_id = 33569
get_synonyms_to_dataframe(model=loadedModel, product_id=required_product_id, num_synonyms=3).show(truncate=False)
"""
 similar for (68570) Диротон таб.20мг №28 738
+----------+----------+------------------------------------------------------------------+
|product_id|similarity|name                                                              |
+----------+----------+------------------------------------------------------------------+
|52119     |0.771674  |(112207) Метопролол ретард-Акрихин таб.пролонг.п.п.о.100мг №30 738|
|60972     |0.771113  |(70768) Амлодипин тб 10мг N20 738                                 |
|137421    |0.768555  |(88567) Пектрол табл. 40 мг. №30 738                              |
+----------+----------+------------------------------------------------------------------+
"""
get_synonyms_to_list(model=loadedModel, product_id=required_product_id, num_synonyms=3)
""" [52119, 60972, 137421] """
