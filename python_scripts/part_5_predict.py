"""
/spark2.4/bin/pyspark
"""
from pyspark.shell import sc
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
# Load model
model = MatrixFactorizationModel.load(sc, "ml_models/myCollaborativeFilter")


# Предсказание
def custom_predictions(user_id):
    return [i.product for i in model.recommendProducts(user=user_id, num=5)]


custom_predictions(user_id=886991)
