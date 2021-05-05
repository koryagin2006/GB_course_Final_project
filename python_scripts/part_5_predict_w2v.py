"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"

# Загрузка модели
loadedWord2Vec = Word2Vec.load(path='ml_models/word2vec_2021_05_05_by_prod_id')
loadedModel = Word2VecModel.load(path='ml_models/word2vec-model_2021_05_05_by_prod_id')

# Create DF
products = spark \
    .read.parquet(user_path + "input_csv_for_recommend_system/Product_dict.parquet") \
    .withColumnRenamed(existing='__index_level_0__', new='product_id')


def my_synonyms(model, word, num_synonyms=5, ):
    name = products.where(condition=F.col('product_id') == required_product_id).select('name').collect()[0]['name']
    print('\n similar for %s' % name)
    return model \
        .findSynonyms(word=str(word), num=num_synonyms) \
        .select('word').withColumnRenamed(existing='word', new='product_id') \
        .join(other=products, on='product_id', how='inner')


# model.getVectors().show(n=30, truncate=False)

required_product_id = 44530
my_synonyms(model=loadedModel, word=required_product_id, num_synonyms=3).show(truncate=False)
