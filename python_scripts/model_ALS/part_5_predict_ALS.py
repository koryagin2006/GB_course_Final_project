"""
/spark2.4/bin/pyspark
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel
from pprint import pprint
import time

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
products = spark.read \
    .format("org.apache.spark.sql.cassandra").options(table="products", keyspace="final_project").load() \
    .withColumn('name', F.regexp_replace('name', r'(\(\d+\) )', ''))


class ModelALS:
    def __init__(self):
        self.model = None
        self.user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
    #
    def load_model(self, model_path):
        """ Загрузка модели из hdfs """
        self.model = ALSModel.load(path=self.user_path + model_path)
    #
    def predict_to_dict(self, user_id, n_recs=5):
        start = time.time()
        preds_dict = {}
        recs_df = self.model \
            .recommendForAllUsers(numItems=n_recs) \
            .where(condition=F.col('user_id') == user_id) \
            .withColumn(colName="rec_exp", col=F.explode("recommendations")) \
            .select(F.col("rec_exp.item_id"))
        #
        preds_dict['user_id'] = user_id
        preds_dict['recommendations'] = [int(row.item_id) for row in recs_df.collect()]
        preds_dict['prediction time'] = round(number=time.time() - start, ndigits=3)
        return preds_dict


model_als = ModelALS()
model_als.load_model(model_path="ml_models/my_als_model_2021-05-11_last_15_weeks.model_als")

predict_als = model_als.predict_to_dict(user_id=471, n_recs=3)
pprint(predict_als)

