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
def custom_predictions(user_id, num_preds=5):
    """
    Предсказание
    :param user_id: id покупателя
    :param num_preds: количество рекомендуемых товаров
    :return: отсортированный список из рекомендуемых товаров
    """
    return [i.product for i in model.recommendProducts(user=user_id, num=num_preds)]


user_list = [903455, 983545, 1962163, 795400, 850395]

for user in user_list:
    print('for user {} predictions list: {}'.format(user, custom_predictions(user_id=user, num_preds=5)))