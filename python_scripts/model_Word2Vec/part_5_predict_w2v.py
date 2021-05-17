"""
/spark2.4/bin/pyspark --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.2
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2VecModel
from pprint import pprint
import time


spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
products = spark.read \
    .format("org.apache.spark.sql.cassandra").options(table="products", keyspace="final_project").load() \
    .withColumn('name', F.regexp_replace('name', r'(\(\d+\) )', ''))


class ModelWord2Vec:
    def __init__(self):
        self.model = None
        self.user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
    #
    def load_model(self, model_path):
        """ Загрузка модели из hdfs """
        self.model = Word2VecModel.load(path=self.user_path + model_path)
    #
    def predict_to_dict(self, product_id, n_recs=5):
        """ Выдача предскааний в виде словаря """
        start = time.time()
        preds_dict = {}
        recs_df = self.model \
            .findSynonyms(word=str(product_id), num=n_recs) \
            .withColumnRenamed(existing='word', new='product_id') \
            .orderBy('similarity', ascending=False)
        #
        preds_dict['product_id'] = product_id
        preds_dict['recommendations'] = [int(row.product_id) for row in recs_df.collect()]
        preds_dict['prediction time'] = round(number=time.time() - start, ndigits=3)
        return preds_dict
    #
    def get_name_product_id(self, products_df, product_id):
        name = products_df.where(condition=F.col('product_id') == product_id).select('name').collect()[0]['name']
        return name
    #
    def predict_to_df(self, products_df, product_id, num_recs=5):
        return self.model \
            .findSynonyms(word=str(product_id), num=num_recs) \
            .withColumnRenamed(existing='word', new='product_id') \
            .join(other=products_df, on='product_id', how='inner') \
            .orderBy('similarity', ascending=False).withColumn('similarity', F.round('similarity', 6)) \
            .select('product_id', 'name')


model_w2v = ModelWord2Vec()
model_w2v.load_model(model_path='ml_models/word2vec_model_2021_05_11')

predict_w2v = model_w2v.predict_to_dict(product_id=33569, n_recs=3)
predict_w2v_df = model_w2v.predict_to_df(products_df=products, product_id=33569, num_recs=3)
product_name = model_w2v.get_name_product_id(products_df=products, product_id=33569)

print(product_name)
pprint(predict_w2v)
predict_w2v_df.show(truncate=False)